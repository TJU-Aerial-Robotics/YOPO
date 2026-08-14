import math
import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from config.config import cfg


class SafetyLoss(nn.Module):

    def __init__(self):
        super(SafetyLoss, self).__init__()
        self.traj_num = cfg['traj_num']

        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        # Exponential barrier cut to 0 at d_safe (continuous); see cost_function.
        self.d0 = float(cfg["d0"])
        self.r = float(cfg["r"])
        self.d_safe = float(cfg["d_safe"])
        self._cost_offset = math.exp(-(self.d_safe - self.d0) / self.r)   # value of exp() at d_safe
        # LogSumExp sharpness for the per-trajectory safety cost: β→0 = mean, β→∞ = max. β≈4 stays
        # stable while putting ~99% of the gradient on the worst sample — see docs/inner_at_bottleneck.md.
        self.safety_beta = float(cfg["safety_beta"])

        # SDF
        self.voxel_size = 0.2
        self.min_bounds = None  # shape: (N, 3)
        self.sdf_shapes = None  # shape: (N, 3)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print("Building ESDF map...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.sdf_maps = self.get_sdf_from_ply(data_dir)
        print("Map built!")

    def forward(self, pos_samples, map_id):
        """
        Args:
            pos_samples: (B*traj_num, S, 3) — positions in world frame, already sampled along trajectory.
            map_id:      (B,)                — which ESDF map to query per real batch element.
        Returns:
            cost_colli:  (B*traj_num,)       — per-trajectory safety cost (see aggregation below).
            collided:    (B*traj_num,)        — bool collision flag, obstacle hits and tunneling.
            dist:        (B*traj_num, S)       — per-sample SDF (detached); YOPOLoss takes the
                                                safety-radius labels and MinDist out of it.
        See _collision_certificate and _aggregate_cost for the detection/aggregation rationale.
        """
        BN, S, _ = pos_samples.shape
        B = BN // self.traj_num

        # 1. per-sample barrier cost + SDF (lump traj_num trajectories into one map query per batch element)
        cost, dist = self.get_distance_cost(pos_samples.reshape(B, self.traj_num * S, 3), map_id)
        cost, dist = cost.reshape(BN, S), dist.reshape(BN, S)

        # 2. continuous collision via the free-ball chain certificate
        collided, first_fail = self._collision_certificate(pos_samples, dist)

        # 3. hybrid cost aggregation
        cost_colli = self._aggregate_cost(cost, first_fail, collided)

        return cost_colli, collided, dist.detach()

    def _collision_certificate(self, pos_samples, dist):
        """Free-ball chain certificate: a sample fails if it is inside an obstacle (SDF ≤ 0) OR the
        segment entering it is uncertified (‖seg‖ > SDF(p_i)+SDF(p_{i+1}); SDF is 1-Lipschitz, so
        the two free balls cover a certified segment). Catches tunneling that point-only checks miss.
        Returns (collided (BN,), first_fail (BN,) index of the first failing sample)."""
        seg_len = (pos_samples[:, 1:] - pos_samples[:, :-1]).norm(dim=-1)           # (BN, S-1)
        gap = seg_len - dist[:, :-1] - dist[:, 1:]                                  # >0 ⇒ uncertified
        fail = dist <= 0                                                            # (BN, S)
        fail[:, 1:] = fail[:, 1:] | (gap > 0)
        return fail.any(dim=-1), th.argmax(fail.float(), dim=1)

    def _aggregate_cost(self, cost, first_fail, collided):
        """Hybrid cost so the gradient always pushes to RETREAT: no collision → smooth log-sum-exp over
        all samples; collision → the sample just BEFORE the first failure (last certified-safe point,
        its ∇SDF points back to free space)."""
        S = cost.shape[1]
        beta = self.safety_beta
        soft = (th.logsumexp(beta * cost, dim=-1) - math.log(S)) / beta            # (BN,) non-colliding
        before = (first_fail - 1).clamp(min=0)
        hard = cost.gather(1, before[:, None]).squeeze(1)                          # (BN,) colliding
        return th.where(collided, hard, soft)

    def get_distance_cost(self, pos, map_id):
        """Trilinear-sample the ESDF at pos (B, N, 3 world) → (cost, sdf), each (B, N).
        grid_sample cost is O(query points), independent of map size, so we query each map's full ESDF directly
        (no pre-cropping to the same size). Looped per distinct map, not per sample — usually one map → one call."""
        own = self.sdf_shapes[map_id].unsqueeze(1)                             # (B,1,3) map dims (x,y,z)
        grid = (pos - self.min_bounds[map_id].unsqueeze(1)) / self.voxel_size  # (B,N,3) voxel coords
        grid = (2.0 * grid / (own - 1.0) - 1.0).clamp(-0.99, 0.99)             # → [-1, 1]
        dist = pos.new_empty(pos.shape[:2])                                    # (B, N)
        for mid in map_id.unique().tolist():
            sel = (map_id == mid).nonzero(as_tuple=True)[0]                    # batch rows on this map
            dist[sel] = F.grid_sample(self.sdf_maps[mid], grid[sel].reshape(1, 1, 1, -1, 3),
                                      mode='bilinear', padding_mode='zeros', align_corners=True).view(len(sel), -1)
        return self.cost_function(dist), dist

    def cost_function(self, d):
        # Exponential barrier cut to 0 at d_safe (continuous):
        #   d < d_safe  → exp(-(d-d0)/r) − exp(-(d_safe-d0)/r)   (exp penalty, shifted to 0 at d_safe)
        #   d ≥ d_safe  → 0                                       (free space contributes no gradient)
        return F.relu(th.exp(-(d - self.d0) / self.r) - self._cost_offset)

    def get_sdf_from_ply(self, path):
        """Build one signed distance field per point-cloud map: voxelize the cloud into an occupancy
        grid (bounds expanded by map_expand_*), then EDT outside − EDT inside → signed distance (m)."""
        sorted_files = self.read_sorted_ply_files(path)
        sdf_maps = []
        min_bounds, sdf_shapes = [], []

        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            print(f"    {os.path.basename(file)}: x=({min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}), "
                  f"y=({min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}), "
                  f"z=({min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f})")

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = th.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2).to(self.device)

            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])
            min_bounds.append(min_bound)

        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()
        return sdf_maps

    def read_sorted_ply_files(self, path):
        """Return the pointcloud-*.ply paths under `path`, sorted by their numeric index."""
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))

        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace('pointcloud-', '').replace('.ply', '')
            return int(number_part)

        return sorted(ply_files, key=extract_index)

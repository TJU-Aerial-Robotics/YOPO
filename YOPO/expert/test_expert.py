"""
Test the cost design with an expert optimizer: optimize trajectories INSIDE the network's
action space and see what the training cost actually prefers.

Variables are the 14 raw per-anchor params (tanh-bounded, decoded through
pred_to_traj_params), scored by YOPOLoss itself — same anchor cones, bounds and weights as
training, plus the inner-centring regularizer in the objective but not in the ranked cost.

Run from the YOPO directory with roscore + rviz up; publishes to /expert_c/*:
    python expert/expert_constrained.py --id 5                # raw dataset state (as trained)
    python expert/expert_constrained.py --id 5 --fixed-state  # canonical state/goal, as test_expert_ros
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # YOPO root
from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.state_transform import StateTransform, transform_body2world, rotate_body2world


class ConstrainedExpert:

    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.traj_num = int(cfg["traj_num"])
        self.V = int(cfg["vertical_num"])
        self.H = int(cfg["horizon_num"])
        self.state_transform = StateTransform()
        self.inner_reg_weight = float(cfg["w_inner_reg"])
        self.goal_length = float(cfg["goal_length"])
        # Exact training cost.
        self.yopo_loss = YOPOLoss()

    def _decode(self, logits, head_pos_w, rot_wb):
        """(N,14) logits -> tanh -> pred_to_traj_params -> world-frame trajectory params."""
        N = self.traj_num
        pred = torch.tanh(logits)
        # inverse of the (B, 14, V, H) -> (B, N, 14) reshape in pred_to_traj_params
        endstate = pred.view(self.V, self.H, 14).permute(2, 0, 1).reshape(1, 14, self.V, self.H)
        inner_pos_b, tail_pva_b, durations, inner_radio_offset = \
            self.state_transform.pred_to_traj_params(endstate)
        inner_pos_b = inner_pos_b.reshape(N, 3)
        tail_pva_b = tail_pva_b.reshape(N, 3, 3)
        durations = durations.reshape(N, 2)
        inner_radio_offset = inner_radio_offset.reshape(N)

        rot = rot_wb.unsqueeze(0).expand(N, 3, 3)
        pos = head_pos_w.unsqueeze(0).expand(N, 3)
        inner_pos_w = transform_body2world(rot, pos, inner_pos_b)
        tail_pos_w = transform_body2world(rot, pos, tail_pva_b[:, 0])
        tail_vel_w = rotate_body2world(rot, tail_pva_b[:, 1])
        tail_acc_w = rotate_body2world(rot, tail_pva_b[:, 2])
        tail_pva_w = torch.stack([tail_pos_w, tail_vel_w, tail_acc_w], dim=1)
        return inner_pos_b, inner_pos_w, tail_pva_w, durations, inner_radio_offset

    @torch.enable_grad()
    def optimize(self, head_pva_w, goal_w, rot_wb, map_id, steps=300, lr=0.05):
        """
        Args:
            head_pva_w: (3,3) world — [pos; vel; acc] rows (current state, fixed).
            goal_w:     (3,)  world goal.
            rot_wb:     (3,3) body→world rotation.
            map_id:     int   ESDF map index.
        Returns dict (CPU): inner, tail_pva, durations, cost, breakdown, collided,
            positions (N,K,3), best_id, head, goal.
        """
        N = self.traj_num
        head = head_pva_w.to(self.device)
        head_N = head.unsqueeze(0).expand(N, 3, 3).contiguous()
        goal_N = goal_w.to(self.device).unsqueeze(0).expand(N, 3).contiguous()
        rot_wb = rot_wb.to(self.device)
        map_id_t = torch.tensor([int(map_id)], device=self.device)

        # Inner-centring target, same as the trainer: normalized goal distance
        # d_goal = min(|goal|, goal_length)/goal_length → dr_target = (d_goal-1) ∈ [-1, 0].
        d_goal = (goal_N[0] - head[0]).norm().clamp(max=self.goal_length) / self.goal_length
        dr_target = d_goal - 1.0

        # Neutral seed per anchor: tanh(0)=0 → inner centred in its bin, tail straight ahead
        # at mid radio, durations = pd_init, zero tail vel/acc.
        logits = torch.zeros(N, 14, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=lr)

        for _ in range(steps):
            inner_b, inner_w, tail_pva_w, durations, dr = self._decode(logits, head[0], rot_wb)
            costs, aux = self.yopo_loss(head_N, tail_pva_w, inner_w, goal_N, map_id_t, durations=durations)
            total = sum(costs.values())                           # (N,) same as trainer total_cost
            # inner regularizer: in the objective (non-collided only) but not in the ranked cost
            inner_reg = self.inner_reg_weight * (dr - dr_target).pow(2) * (~aux["collided"]).float()
            opt.zero_grad()
            (total + inner_reg).sum().backward()
            opt.step()

        with torch.no_grad():
            inner_b, inner_w, tail_pva_w, durations, dr = self._decode(logits, head[0], rot_wb)
            costs, aux = self.yopo_loss(head_N, tail_pva_w, inner_w, goal_N, map_id_t, durations=durations)
            total = sum(costs.values())
            collided = aux["collided"]
            breakdown = {name: c.cpu() for name, c in costs.items()}
            breakdown["InnerReg"] = (self.inner_reg_weight * (dr - dr_target).pow(2)
                                     * (~collided).float()).cpu()   # objective-only, not in cost
            # minco still holds the final coefficients — dense samples for viz
            positions = self.yopo_loss.minco.get_trajectory().get_pos_multi(20)   # (N,K,3)

            # best feasible traj; if every candidate collides the +1e6 offset is uniform → lowest-cost one
            rank = total.clone()
            rank[collided] += 1e6
            best_id = int(rank.argmin())

        return {
            "head": head.cpu(),
            "goal": goal_N[0].cpu(),
            "inner": inner_w.cpu(),                # (N,3) world
            "tail_pva": tail_pva_w.cpu(),          # (N,3,3) world
            "durations": durations.cpu(),          # (N,2)
            "cost": total.cpu(),                   # (N,) training total cost
            "breakdown": breakdown,                # dict name -> (N,) weighted cost
            "collided": collided.cpu(),            # (N,) bool
            "positions": positions.cpu(),          # (N,K,3)
            "best_id": best_id,
        }


def load_sample_full(idx, fixed_state=False):
    """
    Same natural-order sample indexing as test_expert_ros, but also returns the depth image
    (needed by the direction cost) and, by default, keeps the RAW dataset state — i.e. exactly
    what training saw. --fixed-state overrides to a canonical scenario: body vel [10,0,0],
    zero acc, goal 10 m straight ahead of the body.
    """
    from expert.test_expert_ros import _natural_dataset
    dataset = _natural_dataset()
    n = len(dataset)
    if not (0 <= idx < n):
        raise SystemExit(f"--id {idx} out of range [0, {n})")
    np.random.seed(idx)
    image, pos, rot_wb, obs, map_id = dataset[idx]
    pos = np.asarray(pos, dtype=np.float32)
    rot_wb = np.asarray(rot_wb, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)             # [vel_b(3), acc_b(3), goal_b(3)]

    if fixed_state:
        vel_b = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        acc_b = np.zeros(3, dtype=np.float32)
    else:
        vel_b, acc_b = obs[0:3], obs[3:6]

    # body → world, same convention as the trainer
    vel_w = rot_wb @ vel_b
    acc_w = rot_wb @ acc_b
    if fixed_state:
        goal_w = pos + rot_wb @ np.array([10.0, 0.0, 0.0], dtype=np.float32)  # 10 m straight ahead
    else:
        goal_w = pos + rot_wb @ obs[6:9]
    head_pva_w = np.stack([pos, vel_w, acc_w], axis=0).astype(np.float32)    # (3,3)
    depth = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)  # (1,1,H,W)
    return head_pva_w, goal_w.astype(np.float32), rot_wb, int(map_id), depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=0, help="sample index in natural dataset order")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fixed-state", action="store_true",
                    help="reproduce test_expert_ros's state/goal overrides instead of the raw dataset state")
    args = ap.parse_args()

    import rospy
    from sensor_msgs.msg import PointCloud2, Image
    from visualization_msgs.msg import MarkerArray
    from expert.test_expert_ros import load_local_cloud, cloud_xyz, \
        best_traj_markers, start_goal_markers, all_traj_markers, \
        inner_depth_image, world_to_body

    rospy.init_node("expert_constrained_viz", anonymous=False)
    pub_cloud = rospy.Publisher("/expert_c/local_cloud", PointCloud2, queue_size=1, latch=True)
    pub_all_mk = rospy.Publisher("/expert_c/all_trajs_mk", MarkerArray, queue_size=1, latch=True)
    pub_img = rospy.Publisher("/expert_c/inner_depth_img", Image, queue_size=1, latch=True)
    pub_best = rospy.Publisher("/expert_c/best_traj", MarkerArray, queue_size=1, latch=True)
    pub_sg = rospy.Publisher("/expert_c/start_goal", MarkerArray, queue_size=1, latch=True)

    head_pva_w, goal_w, rot_wb, map_id, depth = load_sample_full(args.id, args.fixed_state)
    print(f"[sample {args.id}] map={map_id}  start={head_pva_w[0]}  vel={head_pva_w[1]}  "
          f"goal={goal_w}  (|goal-start|={np.linalg.norm(goal_w - head_pva_w[0]):.1f}m)  "
          f"state={'fixed' if args.fixed_state else 'raw dataset'}")

    expert = ConstrainedExpert()
    res = expert.optimize(torch.from_numpy(head_pva_w), torch.from_numpy(goal_w),
                          torch.from_numpy(rot_wb), map_id, steps=args.steps, lr=args.lr)

    cost = res["cost"].numpy()
    collided = res["collided"].numpy()
    bid = res["best_id"]
    print(f"  anchors={len(cost)}  collision-free={int((~collided).sum())}/{len(cost)}  "
          f"best_id={bid}  best_cost={cost[bid]:.3f}  "
          f"best_time={float(res['durations'][bid].sum()):.2f}s  best_collided={bool(collided[bid])}")
    print("  best-traj cost breakdown (training weights):")
    for name, c in res["breakdown"].items():
        print(f"    {name:<10} {float(c[bid]):8.3f}")

    # per-anchor total duration, laid out like the trajectory grid (V rows × H cols,
    # image order — same arrangement as the anchors in the depth image)
    V, H = int(cfg["vertical_num"]), int(cfg["horizon_num"])
    dur_grid = res["durations"].sum(dim=1).reshape(-1, V, H).numpy()      # (1, V, H)
    print("  per-anchor total duration [s] (grid = anchor layout in image):")
    for h in range(dur_grid.shape[0]):
        if dur_grid.shape[0] > 1:
            print(f"    radio bin {h}:")
        for i in range(V):
            print("    " + "  ".join(f"{dur_grid[h, i, j]:5.2f}" for j in range(H)))

    cloud = load_local_cloud(map_id, head_pva_w[0])
    pos = res["positions"].numpy()                       # (N,K,3)
    inner_b_all = world_to_body(res["inner"].numpy(), head_pva_w[0], rot_wb)
    depth_hw = depth[0, 0].numpy()                       # (H,W) in [0,1]

    rate = rospy.Rate(2)
    print("Publishing to /expert_c/* (frame 'world'). Ctrl-C to stop.")
    while not rospy.is_shutdown():
        pub_cloud.publish(cloud_xyz(cloud))
        pub_all_mk.publish(all_traj_markers(pos, cost, collided, ns="expert_c_all",
                                            inner=res["inner"].numpy()))
        pub_img.publish(inner_depth_image(depth_hw, inner_b_all, cost, collided))
        pub_best.publish(best_traj_markers(pos[bid], res["inner"][bid].numpy(),
                                           res["tail_pva"][bid, 0].numpy()))
        pub_sg.publish(start_goal_markers(head_pva_w[0], goal_w, rot_wb))
        rate.sleep()


if __name__ == "__main__":
    main()

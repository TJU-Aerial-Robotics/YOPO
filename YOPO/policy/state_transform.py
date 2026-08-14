import math
import torch
import numpy as np
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg['goal_length']

        # tail decoding bounds (body frame, free w.r.t. grid)
        self.tail_yaw_half = math.radians(cfg['horizon_camera_fov']) / 2.0
        self.tail_pitch_half = math.radians(cfg['vertical_camera_fov']) / 2.0
        self.tail_radio_max = float(cfg['tail_radio_max'])

        # lower bound on each piece duration, guards A_inv against singularity under extreme tanh outputs
        self.duration_min = 0.1

        # ---- Cache image-order lattice arrays (constant after init) ----
        lp = self.lattice_primitive
        V_H = lp.vertical_num * lp.horizon_num
        self._yaw_grid_img = lp.to_image_order(lp.lattice_angle_node[:, 0])              # (N,)
        self._pitch_grid_img = lp.to_image_order(lp.lattice_angle_node[:, 1])            # (N,)
        self._radio_max_img = lp.to_image_order(lp.lattice_radio_max_node)               # (N,)
        self._pd_init_img = lp.to_image_order(lp.lattice_pd_init_node)                   # (N, 2)
        # Rbp depends only on (i, j) and is h-invariant; take the first V*H entries after reorder.
        self._Rbp_image_grid = lp.to_image_order(lp.getRotation())[:V_H]                 # (V*H, 3, 3)

        # Same constants on CPU, for the *_cpu methods: a per-frame .cpu() is a device sync
        # that stalls while the GPU is busy. Lattice (primitive) order, indexed by lattice_id.
        self._yaw_lattice_np = lp.lattice_angle_node[:, 0].cpu().numpy()                 # (N,)
        self._pitch_lattice_np = lp.lattice_angle_node[:, 1].cpu().numpy()               # (N,)
        self._radio_max_lattice_np = lp.lattice_radio_max_node.cpu().numpy()             # (N,)
        self._pd_init_lattice_np = lp.lattice_pd_init_node.cpu().numpy()                 # (N, 2)
        self._Rbp_image_grid_np = self._Rbp_image_grid.cpu().numpy()                     # (V*H, 3, 3)
        self.lattice_pos_np = lp.lattice_pos_node.cpu().numpy()                          # (N, 3)

    def pred_to_traj_params(self, endstate_pred: torch.Tensor):
        """
        Decode raw network output (B, 14, V, H) into (inner_pos, tail_pva, durations).

        Per (i, j) cell, the 14 traj params are:
            k=0..2   inner waypoint:  (yaw_off, pitch_off, radio)  — grid-anchored
            k=3..5   tail position:   (yaw, pitch, radio)          — body-frame, FOV-bounded
            k=6..8   tail velocity    (body frame)
            k=9..11  tail acceleration (body frame)
            k=12..13 piece duration logits → (tanh+1) · pd_init_per_grid, clamped at duration_min

        Returns:
            inner_pos_b: (B, N, 3)
            tail_pva_b:  (B, N, 3, 3)   — dim -2: 0=pos, 1=vel, 2=acc
            durations:   (B, N, 2)
            inner_radio_offset: (B, N) — raw tanh output for the inner radio (param k=2), in [-1, 1].
                                 0 ⇒ radio_in sits at the midpoint of [0, radio_max] (centred).
        """
        B, _, V, H = endstate_pred.shape
        N = self.lattice_primitive.traj_num   # = V * H

        # (B, 14, V, H) → (B, V, H, 14) → (B, N, 14), image order (i then j); the cached lattice
        # arrays are already in image order, so they line up element-wise.
        pred = endstate_pred.view(B, 14, V, H).permute(0, 2, 3, 1).reshape(B, N, 14)

        # ----- inner waypoint (body frame, grid-anchored) -----
        yaw_grid = self._yaw_grid_img[None].expand(B, -1)
        pitch_grid = self._pitch_grid_img[None].expand(B, -1)
        radio_max_b = self._radio_max_img[None].expand(B, -1)

        delta_yaw = pred[:, :, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = pred[:, :, 1] * self.lattice_primitive.pitch_diff
        # (tanh+1)/2 maps the logit to the per-grid radio range [0, radio_max[k]].
        radio_in = (pred[:, :, 2] + 1.0) * 0.5 * radio_max_b

        yaw_in = yaw_grid + delta_yaw
        pitch_in = pitch_grid + delta_pitch
        cosp = torch.cos(pitch_in)
        inner_x = cosp * torch.cos(yaw_in) * radio_in
        inner_y = cosp * torch.sin(yaw_in) * radio_in
        inner_z = torch.sin(pitch_in) * radio_in
        inner_pos_b = torch.stack([inner_x, inner_y, inner_z], dim=-1)  # (B, N, 3)

        # ----- tail pos (body frame, free, FOV-bounded) -----
        yaw_t = pred[:, :, 3] * self.tail_yaw_half
        pitch_t = pred[:, :, 4] * self.tail_pitch_half
        radio_t = (pred[:, :, 5] + 1.0) * 0.5 * self.tail_radio_max
        cosp_t = torch.cos(pitch_t)
        tail_x = cosp_t * torch.cos(yaw_t) * radio_t
        tail_y = cosp_t * torch.sin(yaw_t) * radio_t
        tail_z = torch.sin(pitch_t) * radio_t
        tail_pos_b = torch.stack([tail_x, tail_y, tail_z], dim=-1)

        # ----- tail vel / acc (body frame, scaled by max) -----
        tail_vel_b = pred[:, :, 6:9] * self.lattice_primitive.vel_max
        tail_acc_b = pred[:, :, 9:12] * self.lattice_primitive.acc_max
        tail_pva_b = torch.stack([tail_pos_b, tail_vel_b, tail_acc_b], dim=-2)

        # ----- piece durations: element-wise (tanh+1) · per-grid pd_init -----
        pd_init_b = self._pd_init_img[None].expand(B, -1, -1)  # (B, N, 2)
        durations = (pred[:, :, 12:14] + 1.0) * pd_init_b
        durations = torch.clamp(durations, min=self.duration_min)

        inner_radio_offset = pred[:, :, 2]  # (B, N) raw tanh Δr; 0 ⇒ centred in [0, radio_max]

        return inner_pos_b, tail_pva_b, durations, inner_radio_offset

    def pred_to_traj_params_cpu(self, endstate_pred: np.ndarray, lattice_id):
        """
        Numpy CPU version used at inference.
        endstate_pred: (M, 14)   selected rows of the per-grid predictions
        lattice_id:    (M,)      lattice indices in PRIMITIVE order

        Returns:
            inner_pos_b: (M, 3)
            tail_pva_b:  (M, 3, 3)  — rows: pos, vel, acc
            durations:   (M, 2)
        """
        lp = self.lattice_primitive

        yaw_grid = self._yaw_lattice_np[lattice_id]
        pitch_grid = self._pitch_lattice_np[lattice_id]
        radio_max = self._radio_max_lattice_np[lattice_id]               # (M,)
        pd_init = self._pd_init_lattice_np[lattice_id]                   # (M, 2)

        # inner
        delta_yaw = endstate_pred[:, 0] * lp.yaw_diff
        delta_pitch = endstate_pred[:, 1] * lp.pitch_diff
        radio_in = (endstate_pred[:, 2] + 1.0) * 0.5 * radio_max

        yaw_in = yaw_grid + delta_yaw
        pitch_in = pitch_grid + delta_pitch
        cosp = np.cos(pitch_in)
        inner_pos_b = np.stack([cosp * np.cos(yaw_in) * radio_in,
                                cosp * np.sin(yaw_in) * radio_in,
                                np.sin(pitch_in) * radio_in], axis=1)

        # tail
        yaw_t = endstate_pred[:, 3] * self.tail_yaw_half
        pitch_t = endstate_pred[:, 4] * self.tail_pitch_half
        radio_t = (endstate_pred[:, 5] + 1.0) * 0.5 * self.tail_radio_max
        cosp_t = np.cos(pitch_t)
        tail_pos_b = np.stack([cosp_t * np.cos(yaw_t) * radio_t,
                               cosp_t * np.sin(yaw_t) * radio_t,
                               np.sin(pitch_t) * radio_t], axis=1)

        tail_vel_b = endstate_pred[:, 6:9] * lp.vel_max
        tail_acc_b = endstate_pred[:, 9:12] * lp.acc_max
        tail_pva_b = np.stack([tail_pos_b, tail_vel_b, tail_acc_b], axis=1)  # (M, 3, 3)

        # durations
        durations = (endstate_pred[:, 12:14] + 1.0) * pd_init
        durations = np.maximum(durations, self.duration_min)
        return inner_pos_b, tail_pva_b, durations

    def prepare_input(self, obs):
        """
        Rotate body-frame obs into per-image-grid primitive frame (V*H rotations, one per (i, j) cell).

        obs: (B, 9)  →  out: (B, 9, V, H)
        """
        B = obs.shape[0]
        V = self.lattice_primitive.vertical_num
        H = self.lattice_primitive.horizon_num
        V_H = V * H

        Rbp_image = self._Rbp_image_grid                                 # (V*H, 3, 3)

        obs = obs.view(B, 3, 3)                                           # (B, 3, 3)
        obs_exp = obs[:, None, :, :].expand(B, V_H, 3, 3)
        Rbp_exp = Rbp_image[None, :, :, :].expand(B, V_H, 3, 3)
        transformed = torch.matmul(obs_exp, Rbp_exp)                      # (B, V*H, 3, 3)

        transformed_flat = transformed.view(B, V_H, 9)
        out = transformed_flat.permute(0, 2, 1).contiguous()              # (B, 9, V*H)
        out = out.view(B, 9, V, H)
        return out

    def prepare_input_cpu(self, obs: np.ndarray) -> np.ndarray:
        """
        Numpy CPU version of prepare_input() used at inference. The tensor is tiny ([B, 9, V, H]),
        so running it on CUDA costs more in kernel launches than the math itself.

        obs: (B, 9)  →  out: (B, 9, V, H)
        """
        B = obs.shape[0]
        V = self.lattice_primitive.vertical_num
        H = self.lattice_primitive.horizon_num
        V_H = V * H

        transformed = np.matmul(obs.reshape(B, 1, 3, 3), self._Rbp_image_grid_np)  # (B, V*H, 3, 3)
        out = transformed.reshape(B, V_H, 9).transpose(0, 2, 1)                    # (B, 9, V*H)
        return np.ascontiguousarray(out).reshape(B, 9, V, H)

    def normalize_obs(self, vel_acc_goal):
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.lattice_primitive.acc_max

        # Clamp the goal direction to unit length
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return vel_acc_goal

    def normalize_obs_cpu(self, vel_acc_goal: np.ndarray) -> np.ndarray:
        """Numpy CPU version of normalize_obs() used at inference. Returns a new array."""
        out = vel_acc_goal.copy()
        out[:, 0:3] /= self.lattice_primitive.vel_max
        out[:, 3:6] /= self.lattice_primitive.acc_max
        goal_norm = np.linalg.norm(out[:, 6:9], axis=1, keepdims=True)
        out[:, 6:9] /= np.maximum(goal_norm, self.goal_length)
        return out


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()

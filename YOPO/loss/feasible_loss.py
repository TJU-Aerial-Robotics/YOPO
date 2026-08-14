import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg


class FeasibleLoss(nn.Module):
    """
    Dynamic-feasibility penalty using discrete sampling.

    (1) Hard limits: peak-violation hinges on speed, acc and jerk (jerk scaled to acc units).
    (2) Heading-gated speed limit (required-curvature): max speed to curve onto the goal
        within the a_soft budget, v_lim = √(a_soft·d / (2·sinθ)); θ = angle(current vel,
        goal), d = goal distance (both fixed inputs, detached), clamped to vel_max.
    (3) Near-goal stop: when ||goal-start|| < 0.9·goal_length, penalize the endpoint speed.
    (4) Curvature speed cap: per-sample v_i ≤ √(a_soft/κ_i), κ detached (gradient on speed only).
    """
    def __init__(self, vel_max: float = None, acc_max: float = None):
        super().__init__()
        self.vel_max = vel_max if vel_max is not None else float(cfg["vel_max_train"])
        self.acc_max = acc_max if acc_max is not None else float(cfg["acc_max_train"])
        self.jerk_max = float(cfg["jerk_max_train"])                    # hard peak-jerk cap (<=0 disables)
        # scale jerk overshoot into acc-equivalent units so it matches the acc hinge's range
        self.jerk_scale = self.acc_max / self.jerk_max if self.jerk_max > 0.0 else 0.0
        # lateral-acc budget, shared by the heading gate (2) and the curvature cap (4)
        self.lat_acc_soft = float(cfg["lat_acc_ratio"]) * self.acc_max
        self.yaw_w = float(cfg["yaw_vlim_weight"])                      # heading gate (2) weight
        self.hover_speed = 0.5                                         # below this speed: no heading cap
        self.goal_length = float(cfg["goal_length"])
        self.stop_thresh = 0.9                                         # near-goal below 0.9 * goal_length
        self.stop_w = float(cfg["tail_stop_weight"])                   # near-goal stop (3) weight
        self.lat_w = float(cfg["lat_acc_weight"])                      # curvature cap (4) weight

    def forward(self, vel_samples, acc_samples, jer_samples=None, head_vel=None, tail_vel=None, goal_dir=None):
        """
        vel_samples/acc_samples/jer_samples: (BN, S, 3) samples along the trajectory.
        head_vel: (BN, 3) current velocity (for gate 2). tail_vel: (BN, 3) endpoint velocity (for stop 3).
        goal_dir: (BN, 3) goal - start. Returns (BN,) per-trajectory penalty.
        """
        # (1) Hard limits (peak-violation hinges).
        speed = vel_samples.norm(dim=-1)                            # (BN, S)
        acc_mag = acc_samples.norm(dim=-1)                          # (BN, S)
        cost = F.relu(acc_mag - self.acc_max).amax(dim=-1)          # (BN,)
        cost = cost + F.relu(speed - self.vel_max).amax(dim=-1)
        if self.jerk_max > 0.0 and jer_samples is not None:
            jer_mag = jer_samples.norm(dim=-1)                      # (BN, S)
            cost = cost + self.jerk_scale * F.relu(jer_mag - self.jerk_max).amax(dim=-1)

        # (2) Curvature speed cap: v_cap = √(a_soft/κ), κ detached.
        if self.lat_w > 0.0:
            with torch.no_grad():
                v_safe = speed.clamp(min=0.5)
                kappa = torch.cross(vel_samples, acc_samples, dim=-1).norm(dim=-1) / v_safe.pow(3)
                v_cap = (self.lat_acc_soft / kappa.clamp(min=1e-6)).sqrt()
                v_cap = v_cap.clamp(min=0.5, max=self.vel_max)
            cost = cost + self.lat_w * F.relu(speed - v_cap).mean(dim=-1)

        # (3) Heading-gated speed limit: v_lim = √(a_soft·d/(2·sinθ)), detached, clamped to vel_max.
        if self.yaw_w > 0.0 and head_vel is not None and goal_dir is not None:
            with torch.no_grad():
                cur_speed = head_vel.norm(dim=-1)
                vh = F.normalize(head_vel, dim=-1, eps=1e-6)
                gd = F.normalize(goal_dir, dim=-1, eps=1e-6)
                cos = (vh * gd).sum(dim=-1).clamp(-1.0, 1.0)
                sin = (1.0 - cos * cos).clamp(min=0.0).sqrt()
                sin_eff = torch.where(cos >= 0, sin, torch.ones_like(sin))  # goal behind 90° → tightest
                d = goal_dir.norm(dim=-1).clamp(min=0.5)
                kappa_req = 2.0 * sin_eff.clamp(min=1e-3) / d
                v_lim = (self.lat_acc_soft / kappa_req).sqrt().clamp(min=0.5, max=self.vel_max)
                v_lim = torch.where(cur_speed < self.hover_speed, torch.full_like(v_lim, self.vel_max), v_lim)
            cost = cost + self.yaw_w * F.relu(speed - v_lim.unsqueeze(-1)).mean(dim=-1)

        # (4) Near-goal stop: penalize endpoint speed when the goal is near.
        if self.stop_w > 0.0 and tail_vel is not None and goal_dir is not None:
            with torch.no_grad():
                near = (goal_dir.norm(dim=-1) / self.goal_length < self.stop_thresh).float()
            cost = cost + self.stop_w * near * tail_vel.norm(dim=-1)

        return cost

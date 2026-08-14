import torch as th
import torch.nn as nn
from config.config import cfg
from minco import MincoS3NU
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.feasible_loss import FeasibleLoss


class YOPOLoss(nn.Module):
    def __init__(self):
        """2-piece MincoS3NU trajectory cost."""
        super(YOPOLoss, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.piece_num = cfg["piece_num"]
        self.minco = MincoS3NU(piece_num=self.piece_num, device=self.device)

        # Per-component weights (keys match the cost dict returned by forward).
        self.weights = {
            "Smooth":    cfg["ws"],
            "Acc":       cfg["wa"],
            "Safety":    cfg["wc"],
            "Goal":      cfg["wg"],
            "Feasible":  cfg["wf"],
            "Time":      cfg["wt"],
        }

        self.eval_per_piece = 15   # trajectory samples per piece (safety + diagnostics)
        self.radius_num = int(cfg["radius_num"])   # time-uniform safety-corridor labels per trajectory
        self.smoothness_loss = SmoothnessLoss()
        self.safety_loss = SafetyLoss()
        self.goal_loss = GuidanceLoss()
        self.feasible_loss = FeasibleLoss()   # speed + acceleration + peak-jerk hinges + shaping

        print("------ Actual Loss ------")
        for name, w in self.weights.items():
            print(f"| {name:<10} = {w:6.4f} |")
        print("-------------------------")

    def forward(self, head_pva, tail_pva, inner_pos, goal, map_id, durations):
        """
        Evaluate the weighted cost of a batch of MINCO trajectories.

        head_pva / tail_pva: (B*N, 3, 3) world frame, rows [pos; vel; acc].
        inner_pos:  (B*N, 3) inner waypoint (world); durations: (B*N, 2) per-piece durations.

        Returns:
            costs: dict of weighted per-component costs, each (B*N,).
            aux:   {"collided": (B*N,) bool free-ball-chain collision flag (catches tunneling),
                    "stats":    dict of detached physical diagnostics,
                    "radius_dist": (B*N, nr) detached SDF corridor labels (SNAP; see _radius_labels)}.
        """
        self.minco.set_parameters(head_pva, tail_pva, inner_pos.unsqueeze(1), durations=durations)
        samples = self.minco.get_trajectory().sample(num_samples_per_piece=self.eval_per_piece)

        smooth_cost, acc_cost = self.smoothness_loss(self.minco)    # ∫ jerk² dt, ∫ acc² dt
        safety_cost, collided, safety_dist = self.safety_loss(samples["pos"], map_id)
        goal_cost = self.goal_loss(tail_pva[:, 0], goal, head_pva[:, 0], collided)
        feasible_cost = self.feasible_loss(samples["vel"], samples["acc"], jer_samples=samples["jer"],
                                           head_vel=head_pva[:, 1], tail_vel=tail_pva[:, 1],
                                           goal_dir=goal - head_pva[:, 0])
        time_cost = self.minco.durations.sum(dim=-1)

        raw = {"Smooth": smooth_cost, "Acc": acc_cost, "Safety": safety_cost, "Goal": goal_cost,
               "Feasible": feasible_cost, "Time": time_cost}
        costs = {k: self.weights[k] * v for k, v in raw.items()}
        aux = {"collided": collided, "stats": self._traj_stats(samples, safety_dist),
               "radius_dist": self._radius_labels(samples["times"], safety_dist)}
        return costs, aux

    @th.no_grad()
    def _radius_labels(self, times, safety_dist):
        """SNAP corridor labels (B*N, nr): SDF of the safety sample nearest each of radius_num
        time-uniform instants. times/safety_dist: (B*N, S)."""
        nr = self.radius_num
        t_k = th.linspace(1.0 / nr, 1.0, nr, device=times.device) * self.minco.durations.sum(dim=-1, keepdim=True)
        snap = (times.unsqueeze(1) - t_k.unsqueeze(2)).abs().argmin(dim=2)   # (B*N, nr) nearest sample
        return safety_dist.gather(1, snap)

    @th.no_grad()
    def _traj_stats(self, samples, safety_dist):
        """Detached per-trajectory physical diagnostics (logging only)."""
        speed = samples["vel"].norm(dim=-1)                            # (B*N, S)
        M = self.eval_per_piece
        dt = (self.minco.durations / M).repeat_interleave(M, dim=1)    # each sample spans duration/M
        total_dur = self.minco.durations.sum(dim=-1)
        path_length = (speed * dt).sum(dim=-1)
        return {
            "Duration": total_dur,
            "PathLength": path_length,
            "AvgSpeed": path_length / total_dur.clamp(min=1e-6),
            "MaxSpeed": speed.amax(dim=-1),
            "MinDist": safety_dist.amin(dim=-1)
        }

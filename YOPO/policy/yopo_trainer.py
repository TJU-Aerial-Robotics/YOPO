import os
import time
import atexit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import *


class YopoTrainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pairwise ranking loss weight
        self.rank_weight = float(cfg["rank_weight"])
        # Inner-waypoint radius regularizer
        self.inner_reg_weight = float(cfg["w_inner_reg"])
        self.goal_length = float(cfg["goal_length"])  # = tail_radio_max; far-goal clip distance
        # Safety-corridor head: asymmetric Laplace NLL in warp space (see _radius_loss)
        self.radius_weight = float(cfg["w_radius"])
        self.radius_lambda = float(cfg["radius_warp_lambda"])
        self.radius_over_weight = float(cfg["radius_over_weight"])
        self.epoch_i = 0
        self.traj_num = cfg['traj_num']
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # Logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)

        # Network
        print("Loading network...")
        self.policy = YopoNetwork()
        self.policy = self.policy.to(self.device)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        except FileNotFoundError:
            print("Training from scratch")

        # Loss
        self.yopo_loss = YOPOLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=True)
        print("Network Loaded! Loading Dataset...")

        # Dataset (you can adjust num_workers according to your training speed)
        self.train_dataloader = DataLoader(YOPODataset(mode='train'), batch_size=self.batch_size, shuffle=True,
                                           num_workers=8, pin_memory=True)
        self.val_dataloader = DataLoader(YOPODataset(mode='valid'), batch_size=self.batch_size, shuffle=False,
                                         num_workers=8, pin_memory=True)
        print("Dataset Loaded!")

    def train(self, epoch, save_interval=None):
        """Run `epoch` epochs of train + eval, checkpointing every `save_interval` epochs."""
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
                    torch.save(self.policy.state_dict(), policy_path)
            self.progress_log.console.log("Train YOPO Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch: int, total_progress):
        """One training epoch: forward/backward each batch, log running means + the rank PMF chart."""
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        running = defaultdict(list)
        rank_buf = []
        start_time = time.time()
        for step, (depth, pos, rot, obs_b, map_id) in enumerate(self.train_dataloader):
            if depth.shape[0] != self.batch_size:  continue  # batch size == number of env

            self.optimizer.zero_grad()
            loss, metrics, pred_rank = self.forward_and_compute_loss(depth, pos, rot, obs_b, map_id)
            loss.backward()
            self.optimizer.step()

            for k, v in metrics.items():
                running[k].append(v.item())
            rank_buf.append(pred_rank.cpu().numpy())

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                gstep = epoch * len(self.train_dataloader) + step
                means = {k: np.mean(v) for k, v in running.items()}
                self.progress_log.console.log(f"Epoch: {epoch}, MeanTraj: {means['mean_traj_loss']:.3g}, "
                                              f"Batch FPS: {batch_fps:.3g}")
                for k, mv in means.items():
                    self.tensorboard_log.add_scalar(self._train_tag(k), mv, gstep)
                running, start_time = defaultdict(list), time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self._log_rank_pmf("Train/PredBestRank", rank_buf, epoch)
        self.progress_log.remove_task(one_epoch_progress)

    @staticmethod
    def _train_tag(k):
        """Map an internal metric key to its training tensorboard tag (see _diagnostics)."""
        if "/" in k:
            grp, name = k.split("/", 1)
            return {"best": "BestTrajLoss", "mean": "MeanTrajLoss", "perf": "BestTrajPerformance"}[grp] + "/" + name
        return {"mean_traj_loss": "Train/MeanTrajLoss", "rank_loss": "Train/RankLoss",
                "inner_reg_loss": "Train/InnerRegLoss", "radius_loss": "Train/RadiusLoss"}[k]

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        """One eval epoch (no grad): mean traj loss, rank loss, best-traj performance, + the rank PMF."""
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        running = defaultdict(list)
        rank_buf = []
        for step, (depth, pos, rot, obs_b, map_id) in enumerate(self.val_dataloader):  # obs: body frame
            if depth.shape[0] != self.batch_size:  continue  # batch size == num of env

            _, metrics, pred_rank = self.forward_and_compute_loss(depth, pos, rot, obs_b, map_id)
            for k, v in metrics.items():
                running[k].append(v.item())
            rank_buf.append(pred_rank.cpu().numpy())
            self.progress_log.update(one_epoch_progress, advance=1)

        # empty-safe: if the val set has no full batch (all skipped), means→nan instead of KeyError
        def _mean(k):
            return float(np.mean(running[k])) if running[k] else float("nan")
        self.progress_log.console.log(f"Eval: {epoch}, MeanTraj: {_mean('mean_traj_loss'):.3g}")
        self.tensorboard_log.add_scalar("Eval/MeanTrajLoss", _mean("mean_traj_loss"), epoch)
        self.tensorboard_log.add_scalar("Eval/RankLoss", _mean("rank_loss"), epoch)
        for k in running:                                          # best-traj performance → Eval/<metric>
            if k.startswith("perf/"):
                self.tensorboard_log.add_scalar("Eval/" + k[len("perf/"):], _mean(k), epoch)
        self._log_rank_pmf("Eval/PredBestRank", rank_buf, epoch)
        self.progress_log.remove_task(one_epoch_progress)

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, map_id):
        depth, pos, rot, obs_b, map_id = [x.to(self.device) for x in [depth, pos, rot, obs_b, map_id]]

        # 1. network forward + body→world lift
        traj = self._predict_world(depth, pos, rot, obs_b)

        # 2. weighted trajectory costs
        costs, aux = self.yopo_loss(traj["head_w"], traj["tail_w"], traj["inner_w"], traj["goal_w"], map_id,
                                    durations=traj["durations"])
        total_cost = sum(costs.values())

        # 3. training loss: generation + selection (rank CE) + inner-waypoint reg + safety-corridor NLL
        loss, rank_loss, inner_reg_loss = self.loss_aggregation(
            total_cost, traj["score"], aux["collided"], traj["inner_dr"], obs_b)
        radius_loss = self._radius_loss(traj["radius"], aux["radius_dist"])
        loss = loss + self.radius_weight * radius_loss

        # 4. tensorboard diagnostics
        metrics, pred_rank = self._diagnostics(
            costs, total_cost, aux["stats"], traj["score"], rank_loss, inner_reg_loss, radius_loss)
        return loss, metrics, pred_rank

    def _predict_world(self, depth, pos, rot, obs_b):
        """Network forward + body→world lift. Returns per-trajectory (B*N) world-frame
        head/tail/inner PVA and goal, plus durations, the raw score grid, and the raw inner Δr.
        NOTE: policy.inference normalizes obs_b in place."""
        BN = self.batch_size * self.traj_num
        # MUST read the raw obs_b BEFORE inference — policy.inference normalizes obs_b in place.
        goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
        inner_b, tail_pva_b, durations, score, inner_dr, radius = self.policy.inference(depth, obs_b)

        inner_b = inner_b.reshape(BN, 3)
        tail_b = tail_pva_b.reshape(BN, 3, 3)                            # rows: pos, vel, acc
        # expand per-image pos/rot to per-trajectory: (B, ...) → (B*N, ...)
        pos_ex = pos.repeat_interleave(self.traj_num, dim=0)
        rot_ex = rot.repeat_interleave(self.traj_num, dim=0)

        inner_w = transform_body2world(rot_ex, pos_ex, inner_b)
        tail_w = torch.stack([transform_body2world(rot_ex, pos_ex, tail_b[:, 0]),
                              rotate_body2world(rot_ex, tail_b[:, 1]),
                              rotate_body2world(rot_ex, tail_b[:, 2])], dim=1)   # [B*N, 3, 3]
        head_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1).repeat_interleave(self.traj_num, dim=0)
        return {"head_w": head_w, "tail_w": tail_w, "inner_w": inner_w,
                "goal_w": goal_w.repeat_interleave(self.traj_num, dim=0),
                "durations": durations.reshape(BN, 2),
                "score": score, "inner_dr": inner_dr, "radius": radius.reshape(BN, -1)}

    def loss_aggregation(self, total_cost, score, safety_collided, inner_radio_offset, obs_b):
        """Generation + Selection + inner-waypoint regularizer. Returns (loss, rank_loss, inner_reg_loss)."""
        B, N = self.batch_size, self.traj_num
        score_flat = score.reshape(B * N)
        collided = safety_collided.view(B, N)                                # (B, N) bool

        # Generation: score-weighted cost, score DETACHED (weight only, no grad to score). Normalize
        # per image (over its N trajs) so obstacle-dense and open scenes contribute equally.
        cost_2d = total_cost.view(B, N)
        cost_mean = cost_2d.detach().mean(dim=1, keepdim=True).clamp(min=1e-3)
        gen_loss = ((cost_2d / cost_mean).reshape(-1) * score_flat.detach().clamp(min=1e-3)).mean()

        # Selection: top-1 CE over FEASIBLE trajs. Offset colliding trajs above every feasible one so
        # argmin picks the best feasible; a fully colliding image falls back to the least-bad traj.
        cost_det = total_cost.detach().view(B, N)
        rank_loss = F.cross_entropy(score_flat.view(B, N), (cost_det + (cost_det.max() + 1.0) * collided.float()).argmin(dim=1))

        # Inner-waypoint regularizer (centring, feasible trajs only): pull raw Δr toward a goal-adaptive target.
        dr_target = (obs_b[:, 6:9].norm(dim=-1) - 1.0).clamp(-1.0, 0.0)       # (B,) from normalized goal dist
        inner_reg_sq = (inner_radio_offset - dr_target[:, None]).pow(2)       # (B, N)
        inner_reg_loss = inner_reg_sq[~collided].mean() if (~collided).any() else inner_reg_sq.new_zeros(())

        loss = gen_loss + self.rank_weight * rank_loss + self.inner_reg_weight * inner_reg_loss
        return loss, rank_loss, inner_reg_loss

    def _radius_loss(self, radius_pred, radius_dist):
        """Safety-corridor NLL: warp y = 1-exp(-d/λ), L = w·|y-μ|/b + log b (radius_pred = [μ; b]).
        b absorbs unpredictable samples; μ > y (over-estimating clearance) weighs radius_over_weight."""
        nr = radius_dist.shape[1]
        mu, b = radius_pred[:, :nr], radius_pred[:, nr:]
        y = 1.0 - torch.exp(-radius_dist.clamp(min=0.0) / self.radius_lambda)
        err = mu - y
        w = 1.0 + (self.radius_over_weight - 1.0) * (err > 0).detach().float()   # heavier when μ > y
        return (w * err.abs() / b + b.log()).mean()

    def _diagnostics(self, costs, total_cost, traj_stats, score, rank_loss, inner_reg_loss, radius_loss):
        """Detached metrics for tensorboard (internal keys mapped to tags by _train_tag / eval_one_epoch).
        best/* = the selected (max-score) traj per image; mean/* = over all grids; perf/* = best-traj
        performance. Returns (metrics, pred_rank); pred_rank = rank of the selected traj in the true cost order."""
        B, N = self.batch_size, self.traj_num
        best_idx = score.detach().reshape(B, N).argmax(dim=1)                 # (B,)
        def _best(c):  # (B*N,) → per-image best-scoring grid, averaged over batch
            return c.detach().view(B, N).gather(1, best_idx[:, None]).squeeze(1).mean()

        metrics = {}
        for name, c in costs.items():                       # per-component loss: best traj vs mean over all
            metrics[f"best/{name}"] = _best(c)
            metrics[f"mean/{name}"] = c.detach().mean()
        for name, v in traj_stats.items():                  # best-traj performance (Duration/PathLength/AvgSpeed/MaxSpeed/MinDist)
            metrics[f"perf/{name}"] = _best(v)
        metrics["mean_traj_loss"] = total_cost.detach().mean()
        metrics["rank_loss"] = rank_loss.detach()
        metrics["inner_reg_loss"] = inner_reg_loss.detach()
        metrics["radius_loss"] = radius_loss.detach()

        # rank of the predicted-best traj in the true cost order (0 = true optimum)
        cost_2d = total_cost.detach().view(B, N)
        selected = cost_2d.gather(1, best_idx[:, None])
        pred_rank = (cost_2d < selected).sum(dim=1)                           # (B,) 0-based
        return metrics, pred_rank

    def _log_rank_pmf(self, tag, rank_buf, step):
        """Bar chart of the predicted-best traj's cost-rank distribution, normalized to a
        probability (bars sum to 1). rank=0 ⇒ the prediction is the true optimum."""
        if not rank_buf:
            return
        ranks = np.concatenate(rank_buf)
        pmf = np.bincount(ranks, minlength=self.traj_num).astype(float) / ranks.size
        fig, ax = plt.subplots()
        ax.bar(np.arange(self.traj_num), pmf)
        ax.set_xlabel("rank of predicted-best traj (0 = true optimum)")
        ax.set_ylabel("probability")
        ax.set_xticks(np.arange(self.traj_num))
        ax.set_ylim(0, 1)
        self.tensorboard_log.add_figure(tag, fig, step)
        plt.close(fig)

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
            torch.save(self.policy.state_dict(), policy_path)
            atexit.unregister(self._exit_func)

    def get_next_log_path(self, base_path):
        """Return the next unused YOPO_<n> subdir under base_path (auto-incrementing run id)."""
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("YOPO_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"YOPO_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path

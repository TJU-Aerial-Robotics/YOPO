"""
Evaluation of the piecewise-polynomial trajectory produced by MincoS3NU.
Given per-piece durations and coefficients, samples position/velocity/acceleration/jerk for the
whole batch at once (pieces folded into the batch dim, one bmm against a time-power basis).
"""
import torch
from typing import Tuple, List


class Piece:
    """
    A single polynomial piece of degree D (D=5 for MINCO_S3NU).

    Args:
        coeff_mat: Coefficient matrix. Shape (batch, 3, D+1) where
                   coeff_mat[:, :, 0] is constant term (t^0),
                   coeff_mat[:, :, 1] is coefficient for t^1,
                   ..., coeff_mat[:, :, D] is coefficient for t^D.
    """
    def __init__(self, coeff_mat: torch.Tensor):
        self.coeff_mat = coeff_mat            # (B, 3, D+1)
        self.degree = coeff_mat.shape[2] - 1

    def _horner(self, t: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        # coeff: (B, 3, K), t: (B,)
        y = coeff[:, :, -1]
        for i in range(coeff.shape[2] - 2, -1, -1):
            y = y * t.unsqueeze(1) + coeff[:, :, i]
        return y

    def _horner_multi(self, t: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        # coeff: (B, 3, K), t: (B, S)
        y = coeff[:, :, -1].unsqueeze(-1)  # (B, 3, 1)
        for i in range(coeff.shape[2] - 2, -1, -1):
            y = y * t.unsqueeze(1) + coeff[:, :, i].unsqueeze(-1)
        return y  # (B, 3, S)

    def get_pos(self, t: torch.Tensor) -> torch.Tensor:
        return self._horner(t, self.coeff_mat)

    def get_vel(self, t: torch.Tensor) -> torch.Tensor:
        w = torch.arange(1, self.degree + 1, device=t.device, dtype=t.dtype)
        return self._horner(t, self.coeff_mat[:, :, 1:] * w)

    def get_acc(self, t: torch.Tensor) -> torch.Tensor:
        w1 = torch.arange(2, self.degree + 1, device=t.device, dtype=t.dtype)
        w2 = torch.arange(1, self.degree, device=t.device, dtype=t.dtype)
        return self._horner(t, self.coeff_mat[:, :, 2:] * (w1 * w2))

    def get_jer(self, t: torch.Tensor) -> torch.Tensor:
        w1 = torch.arange(3, self.degree + 1, device=t.device, dtype=t.dtype)
        w2 = torch.arange(2, self.degree, device=t.device, dtype=t.dtype)
        w3 = torch.arange(1, self.degree - 1, device=t.device, dtype=t.dtype)
        return self._horner(t, self.coeff_mat[:, :, 3:] * (w1 * w2 * w3))

    def get_pos_multi(self, t: torch.Tensor) -> torch.Tensor:
        return self._horner_multi(t, self.coeff_mat)

    def get_vel_multi(self, t: torch.Tensor) -> torch.Tensor:
        w = torch.arange(1, self.degree + 1, device=t.device, dtype=t.dtype)
        return self._horner_multi(t, self.coeff_mat[:, :, 1:] * w)

    def get_acc_multi(self, t: torch.Tensor) -> torch.Tensor:
        w1 = torch.arange(2, self.degree + 1, device=t.device, dtype=t.dtype)
        w2 = torch.arange(1, self.degree, device=t.device, dtype=t.dtype)
        return self._horner_multi(t, self.coeff_mat[:, :, 2:] * (w1 * w2))

    def get_jer_multi(self, t: torch.Tensor) -> torch.Tensor:
        w1 = torch.arange(3, self.degree + 1, device=t.device, dtype=t.dtype)
        w2 = torch.arange(2, self.degree, device=t.device, dtype=t.dtype)
        w3 = torch.arange(1, self.degree - 1, device=t.device, dtype=t.dtype)
        return self._horner_multi(t, self.coeff_mat[:, :, 3:] * (w1 * w2 * w3))


_DERIV_CACHE = {}


def _deriv_basis(degree: int, device, dtype) -> torch.Tensor:
    """(D+1, 4(D+1)) constant [I | D | D² | D³]: right-multiplying a coefficient matrix by it
    yields the coefficients of the polynomial and of its first three derivatives."""
    key = (degree, device, dtype)
    basis = _DERIV_CACHE.get(key)
    if basis is None:
        K = degree + 1
        D = torch.zeros(K, K, device=device, dtype=dtype)
        for k in range(1, K):
            D[k, k - 1] = k                      # d/dt: c_k t^k -> k c_k t^(k-1)
        D2 = D @ D
        basis = torch.cat([torch.eye(K, device=device, dtype=dtype), D, D2, D2 @ D], dim=1)
        _DERIV_CACHE[key] = basis
    return basis


class Trajectory:
    """
    A multi-piece polynomial trajectory. Per-piece durations may differ per batch
    element and per piece (used when MINCO learns the piece times).

    Args:
        durations: (batch, N) — duration of each piece.
        coeff_mats: List of N coefficient matrices, each (batch, 3, D+1).
    """
    def __init__(self, durations: torch.Tensor, coeff_mats: List[torch.Tensor]):
        self.batch_size = durations.shape[0]
        self.num_pieces = durations.shape[1]
        self.durations = durations

        # cum_durations[:, i] = sum of durations[0..i].
        self.cum_durations = torch.cumsum(durations, dim=1)               # (B, N)
        self.total_duration = self.cum_durations[:, -1]                   # (B,)
        # piece_start[:, i] = start time of piece i within the whole trajectory.
        zeros = torch.zeros(self.batch_size, 1, device=durations.device, dtype=durations.dtype)
        self.piece_start = torch.cat([zeros, self.cum_durations[:, :-1]], dim=1)  # (B, N)

        self.pieces = [Piece(coeff_mats[i]) for i in range(self.num_pieces)]

    def locate_piece(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Locate which piece contains time t and compute relative time within that piece.

        Args:
            t: Scalar tensor or (B,) tensor.
        Returns:
            (piece_indices, relative_times), both (B,).
        """
        if t.dim() == 0:
            t = t.expand(self.batch_size)

        t = torch.clamp(t, torch.zeros_like(self.total_duration), self.total_duration)
        # piece_indices[b] = number of pieces whose end-time <= t[b], clamped.
        piece_indices = (self.cum_durations <= t.unsqueeze(1)).sum(dim=1).clamp(0, self.num_pieces - 1)
        prev_cum = self.piece_start.gather(1, piece_indices.unsqueeze(1)).squeeze(1)
        relative_times = t - prev_cum
        return piece_indices, relative_times

    def _eval(self, t: torch.Tensor, fn: str) -> torch.Tensor:
        """Evaluate trajectory at time t. Per-batch piece index may differ."""
        idx, rel_t = self.locate_piece(t)
        out = None
        for i in range(self.num_pieces):
            val_i = getattr(self.pieces[i], fn)(rel_t)                  # (B, 3)
            mask = (idx == i).to(val_i.dtype).unsqueeze(-1)             # (B, 1)
            out = mask * val_i if out is None else out + mask * val_i
        return out

    def get_pos(self, t: torch.Tensor) -> torch.Tensor:
        return self._eval(t, "get_pos")

    def get_vel(self, t: torch.Tensor) -> torch.Tensor:
        return self._eval(t, "get_vel")

    def get_acc(self, t: torch.Tensor) -> torch.Tensor:
        return self._eval(t, "get_acc")

    def get_jer(self, t: torch.Tensor) -> torch.Tensor:
        return self._eval(t, "get_jer")

    def _get_t_norm(self, num_samples_per_piece: int) -> torch.Tensor:
        """Normalized sample times in (0, 1]."""
        return torch.linspace(
            0, 1, num_samples_per_piece + 1,
            device=self.durations.device,
            dtype=self.durations.dtype
        )[1:]

    def _time_basis(self, num_samples_per_piece: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-piece-uniform sample times, pieces folded into the batch dim.
        Returns (t_rel (B, N, S), powers (B·N, D+1, S) with powers[..., k, s] = t_rel^k)."""
        K = self.pieces[0].degree + 1
        t_norm = self._get_t_norm(num_samples_per_piece)
        t_rel = t_norm.view(1, 1, -1) * self.durations.unsqueeze(-1)               # (B, N, S)
        exps = torch.arange(K, device=t_rel.device, dtype=t_rel.dtype).view(1, K, 1)
        powers = t_rel.reshape(self.batch_size * self.num_pieces, 1, -1) ** exps
        return t_rel, powers

    def _coeff_stack(self) -> torch.Tensor:
        """(B·N, 3, D+1) — per-piece coefficients folded into the batch dim."""
        return torch.stack([p.coeff_mat for p in self.pieces], dim=1).flatten(0, 1)

    def get_pos_multi(self, num_samples_per_piece: int = 10) -> torch.Tensor:
        """
        Sample uniformly inside each piece. Returns (batch, num_pieces * S, 3).
        """
        B, N = self.batch_size, self.num_pieces
        _, powers = self._time_basis(num_samples_per_piece)
        pos = torch.bmm(self._coeff_stack(), powers)                              # (B·N, 3, S)
        return pos.view(B, N, 3, -1).permute(0, 1, 3, 2).reshape(B, -1, 3)

    def sample(self, num_samples_per_piece: int = 10) -> dict:
        """
        Per-piece-uniform sampling for cost computation. One bmm evaluates the polynomial and
        its three derivatives for every piece at once.
        Returns dict with 'times', 'pos', 'vel', 'acc', 'jer'.
            pos/vel/acc/jer: (batch, total_samples, 3)
            times:           (batch, total_samples)
        """
        B, N = self.batch_size, self.num_pieces
        K = self.pieces[0].degree + 1
        t_rel, powers = self._time_basis(num_samples_per_piece)

        # coefficients of [pos; vel; acc; jer] stacked along the row dim: (B·N, 4·3, K)
        basis = _deriv_basis(K - 1, self.durations.device, self.durations.dtype)
        coeff = (self._coeff_stack() @ basis).view(-1, 3, 4, K).permute(0, 2, 1, 3).flatten(1, 2)
        vals = torch.bmm(coeff, powers).view(B, N, 4, 3, -1)                      # (B, N, 4, 3, S)
        out = vals.permute(0, 2, 1, 4, 3).reshape(B, 4, -1, 3)                    # (B, 4, N·S, 3)

        return {
            "times": (self.piece_start.unsqueeze(-1) + t_rel).flatten(1, 2),
            "pos": out[:, 0],
            "vel": out[:, 1],
            "acc": out[:, 2],
            "jer": out[:, 3],
        }

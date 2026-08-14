"""
PyTorch MINCO_S3NU (quintic, s=3, PVA boundary) trajectory solver.
Batched, GPU-friendly, differentiable through the durations.
"""
import torch
import torch.nn as nn
from .trajectory import Trajectory


class MincoS3NU(nn.Module):
    """MINCO_S3NU quintic solver. Time-dependent state (durations, A_inv, coeffs) and the
    batch size are set per call in set_parameters(); only the piece count is fixed here."""
    def __init__(self, piece_num: int, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = torch.float32
        self.N = piece_num
        self.B = None       # batch size, from durations.shape[0]

        # A is sparse with a fixed pattern; every nonzero is const * t_piece^power.
        entries = self._constraint_pattern(piece_num)
        S = 6 * piece_num
        self._A_index = torch.tensor([r * S + c for r, c, _, _, _ in entries], device=self.device)
        self._T_index = torch.tensor([p * 6 + pw for _, _, p, pw, _ in entries], device=self.device)
        self._A_const = torch.tensor([c for *_, c in entries], device=self.device, dtype=self.dtype)
        self._exps = torch.arange(6, device=self.device, dtype=self.dtype)

    @staticmethod
    def _constraint_pattern(N: int) -> list:
        """Nonzeros of the constraint matrix A as (row, col, piece, power, const),
        i.e. A[row, col] = const * durations[piece] ** power."""
        ent = [(0, 0, 0, 0, 1.0), (1, 1, 0, 0, 1.0), (2, 2, 0, 0, 2.0)]   # start conditions
        for i in range(N - 1):
            r = 6 * i
            for p in range(6):                                    # piece end position
                ent.append((r + 3, 6*i + p, i, p, 1.0))
            for p in range(6):                                    # position continuity
                ent.append((r + 4, 6*i + p, i, p, 1.0))
            ent.append((r + 4, 6*(i+1), 0, 0, -1.0))
            for p, c in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):     # velocity continuity
                ent.append((r + 5, 6*i + 1 + p, i, p, c))
            ent.append((r + 5, 6*(i+1) + 1, 0, 0, -1.0))
            for p, c in enumerate([2.0, 6.0, 12.0, 20.0]):        # acceleration continuity
                ent.append((r + 6, 6*i + 2 + p, i, p, c))
            ent.append((r + 6, 6*(i+1) + 2, 0, 0, -2.0))
            for p, c in enumerate([6.0, 24.0, 60.0]):             # jerk continuity
                ent.append((r + 7, 6*i + 3 + p, i, p, c))
            ent.append((r + 7, 6*(i+1) + 3, 0, 0, -6.0))
            for p, c in enumerate([24.0, 120.0]):                 # snap continuity
                ent.append((r + 8, 6*i + 4 + p, i, p, c))
            ent.append((r + 8, 6*(i+1) + 4, 0, 0, -24.0))

        i = N - 1                                                 # end conditions
        r = 6 * i
        for p in range(6):
            ent.append((r + 3, r + p, i, p, 1.0))
        for p, c in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            ent.append((r + 4, r + 1 + p, i, p, c))
        for p, c in enumerate([2.0, 6.0, 12.0, 20.0]):
            ent.append((r + 5, r + 2 + p, i, p, c))
        return ent

    def _build_constraint_matrix(self, t_pow: torch.Tensor) -> torch.Tensor:
        """Scatter the nonzero pattern into A. t_pow: (B, N, 6) powers of the durations."""
        B, S = self.B, 6 * self.N
        values = t_pow.reshape(B, -1)[:, self._T_index] * self._A_const   # (B, nnz)
        A = torch.zeros(B, S * S, device=self.device, dtype=self.dtype)
        A[:, self._A_index] = values
        return A.view(B, S, S)

    def set_parameters(self, head_pva: torch.Tensor,
                       tail_pva: torch.Tensor,
                       inner_pts: torch.Tensor,
                       durations: torch.Tensor) -> None:
        """
        Solve for polynomial coefficients given boundary conditions and waypoints.

        Args:
            head_pva: Start state (batch, 3, 3)
            tail_pva: End state (batch, 3, 3)
            inner_pts: Inner waypoint positions. Shape (batch, N-1, 3).
            durations: Per-piece durations (batch, N). Sets the batch size and rebuilds
                       A and A_inv; gradient flows back to durations.
        """
        durations = durations.to(self.device, dtype=self.dtype)
        assert durations.dim() == 2 and durations.shape[1] == self.N, \
            f"durations shape {tuple(durations.shape)} != (batch, {self.N})"
        self.B = durations.shape[0]
        self.durations = durations
        t_pow = durations.unsqueeze(-1) ** self._exps            # (B, N, 6): [1, T, T², T³, T⁴, T⁵]
        self.T1, self.T2, self.T3, self.T4, self.T5 = (t_pow[..., k] for k in range(1, 6))
        self.A_inv = torch.linalg.inv(self._build_constraint_matrix(t_pow))

        inner_pts = inner_pts.to(self.device, dtype=self.dtype)

        N, B = self.N, self.B

        # b rows: [head pva | (waypoint i, 5 continuity zeros) for each inner point | tail pva]
        zeros5 = torch.zeros(B, 5, 3, device=self.device, dtype=self.dtype)
        blocks = [head_pva.to(self.device, dtype=self.dtype)]
        for i in range(N - 1):
            blocks += [inner_pts[:, i:i+1], zeros5]
        blocks.append(tail_pva.to(self.device, dtype=self.dtype))
        b = torch.cat(blocks, dim=1)

        self.coeffs = torch.bmm(self.A_inv, b)

    def get_trajectory(self) -> Trajectory:
        coeff_mats = [
            self.coeffs[:, 6*i:6*i+6].transpose(1, 2)
            for i in range(self.N)
        ]
        return Trajectory(self.durations, coeff_mats)

    def get_energy(self) -> torch.Tensor:
        """
        Compute trajectory energy: integral of jerk squared.
        For quintic p(t), jerk(t) = 6 c3 + 24 c4 t + 60 c5 t^2;
        the six terms below are the closed-form coefficients of
        sum_pieces ∫_0^T (6 c3 + 24 c4 t + 60 c5 t^2)^2 dt.
        """
        coeffs = self.coeffs.view(self.B, self.N, 6, 3)
        c3, c4, c5 = coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5]

        c3_sq = (c3 ** 2).sum(-1)
        c4_sq = (c4 ** 2).sum(-1)
        c5_sq = (c5 ** 2).sum(-1)
        c3_c4 = (c3 * c4).sum(-1)
        c3_c5 = (c3 * c5).sum(-1)
        c4_c5 = (c4 * c5).sum(-1)

        energy = (
            36.0 * c3_sq * self.T1 +
            144.0 * c3_c4 * self.T2 +
            192.0 * c4_sq * self.T3 +
            240.0 * c3_c5 * self.T3 +
            720.0 * c4_c5 * self.T4 +
            720.0 * c5_sq * self.T5
        )

        return energy.sum(dim=1)

    def get_acc_energy(self) -> torch.Tensor:
        """
        Compute trajectory acceleration energy: integral of acceleration squared.
        For quintic p(t), acc(t) = 2 c2 + 6 c3 t + 12 c4 t^2 + 20 c5 t^3;
        the closed-form sum_pieces ∫_0^T (2 c2 + 6 c3 t + 12 c4 t^2 + 20 c5 t^3)^2 dt.
        Use as a penalty on excessive acceleration (weight wa).
        """
        coeffs = self.coeffs.view(self.B, self.N, 6, 3)
        c2, c3, c4, c5 = coeffs[:, :, 2], coeffs[:, :, 3], coeffs[:, :, 4], coeffs[:, :, 5]

        c2_sq = (c2 ** 2).sum(-1)
        c3_sq = (c3 ** 2).sum(-1)
        c4_sq = (c4 ** 2).sum(-1)
        c5_sq = (c5 ** 2).sum(-1)
        c2_c3 = (c2 * c3).sum(-1)
        c2_c4 = (c2 * c4).sum(-1)
        c2_c5 = (c2 * c5).sum(-1)
        c3_c4 = (c3 * c4).sum(-1)
        c3_c5 = (c3 * c5).sum(-1)
        c4_c5 = (c4 * c5).sum(-1)

        T6 = self.T5 * self.T1
        T7 = T6 * self.T1

        energy = (
            4.0 * c2_sq * self.T1 +
            12.0 * c2_c3 * self.T2 +
            (12.0 * c3_sq + 16.0 * c2_c4) * self.T3 +
            (20.0 * c2_c5 + 36.0 * c3_c4) * self.T4 +
            (28.8 * c4_sq + 48.0 * c3_c5) * self.T5 +
            80.0 * c4_c5 * T6 +
            (400.0 / 7.0) * c5_sq * T7
        )

        return energy.sum(dim=1)

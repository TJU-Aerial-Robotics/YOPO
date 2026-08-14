"""
PyTorch MINCO Trajectory Optimization

MIT License

This is a PyTorch implementation of the MINCO (Minimum Control) trajectory
optimization framework, originally developed by Zhepei Wang in C++.

Features:
- Batch processing for parallel trajectory optimization
- GPU acceleration support
- Automatic differentiation through PyTorch autograd
- Quintic polynomial trajectories (MINCO_S3NU)

Example usage:
    >>> import torch
    >>> from pytorch_minco import MincoS3NU
    >>>
    >>> # Single trajectory
    >>> # Format: (batch, pva, xyz) - dim 1: 0=pos, 1=vel, 2=acc, dim 2: xyz
    >>> head_pva = torch.zeros(1, 3, 3)  # [pos, vel, acc], each is xyz
    >>> tail_pva = torch.zeros(1, 3, 3)
    >>> head_pva[0, 0, :] = [0.0, 0.0, 0.0]  # start position xyz
    >>> tail_pva[0, 0, :] = [1.0, 1.0, 1.0]  # end position xyz
    >>>
    >>> minco = MincoS3NU(piece_num=2)
    >>>
    >>> # inner_pts format: (batch, N-1, 3); durations: (batch, N)
    >>> inner_pts = torch.tensor([[[0.5, 0.5, 0.5]]])  # (1, 1, 3)
    >>> durations = torch.ones(1, 2) * 1.0
    >>>
    >>> minco.set_parameters(head_pva, tail_pva, inner_pts, durations)
    >>> energy = minco.get_energy()
"""

from .minco import MincoS3NU
from .trajectory import Piece, Trajectory

__version__ = "0.1.0"
__all__ = ["MincoS3NU", "Piece", "Trajectory"]

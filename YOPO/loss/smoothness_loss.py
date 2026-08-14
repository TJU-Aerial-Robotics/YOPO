import torch.nn as nn


class SmoothnessLoss(nn.Module):
    """Parameter-free wrapper over MincoS3NU's closed-form energies.
    Returns (jerk_energy, acc_energy) = (∫ jerk² dt, ∫ acc² dt), each (batch,)."""
    def forward(self, minco):
        return minco.get_energy(), minco.get_acc_energy()

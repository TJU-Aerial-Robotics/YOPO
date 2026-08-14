import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidanceLoss(nn.Module):
    """
    Endpoint-goal distance cost, split along the start->goal ray (world frame):
      - parallel: how far the endpoint is from the goal *along* the direction (progress);
      - perpendicular: lateral deviation off the ray, weighted lower (`perp_weight`) so a slightly
        different heading (e.g. to dodge obstacles) is not punished hard.
    """
    def __init__(self, perp_weight=0.3):
        super(GuidanceLoss, self).__init__()
        self.perp_weight = perp_weight

    def forward(self, end_pos, goal, start_pos, collided=None):
        dir_unit = F.normalize(goal - start_pos, dim=-1, eps=1e-6)      # (B*N,3) start->goal unit
        err = end_pos - goal                                            # (B*N,3)
        err_par = (err * dir_unit).sum(dim=-1, keepdim=True) * dir_unit
        err_perp = err - err_par
        par_cost = F.smooth_l1_loss(err_par, torch.zeros_like(err_par), reduction='none').sum(dim=-1)
        perp_cost = F.smooth_l1_loss(err_perp, torch.zeros_like(err_perp), reduction='none').sum(dim=-1)
        cost = par_cost + self.perp_weight * perp_cost
        if collided is not None:
            cost = torch.where(collided, cost.detach(), cost)
        return cost

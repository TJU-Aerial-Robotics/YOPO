"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
import numpy as np
from config.config import cfg
from policy.models.backbone import YopoBackbone
from policy.models.head import YopoHead
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            hidden_state=64,
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Per grid cell: 14 tanh traj params + 1 softplus score + 2·radius_num corridor channels
        # (radius_num sigmoid means μ, then radius_num bounded scales b).
        self.params_per_bin = 14
        self.radius_num = int(cfg["radius_num"])
        self.radius_b_min = float(cfg["radius_b_min"])
        self.radius_b_max = float(cfg["radius_b_max"])
        output_dim = self.params_per_bin + 1 + 2 * self.radius_num

        self.image_backbone = YopoBackbone(hidden_state)
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network.

            Conv output layout: (B, 15 + 2·nr, V, H). Channels 0..13 → tanh (traj params
            below), 14 → softplus (score), 15:15+nr → sigmoid (warped safety radii μ =
            1-exp(-d/λ)), 15+nr:15+2nr → bounded-sigmoid Laplace scales b (confidence):

              [0:3]   inner waypoint offset  (yaw_off, pitch_off, radio_within_bin)
              [3:6]   tail direction         (yaw, pitch, radio) in body frame
              [6:9]   tail velocity          (body frame)
              [9:12]  tail acceleration      (body frame)
              [12:14] piece duration logits  → element-wise (tanh+1) · pd_init_per_grid
              [14]    score

            Returns:
                endstate: (B, 14, V, H)   — tanh-bounded trajectory params
                score:    (B, V, H)       — softplus score per cell
                radius:   (B, 2·nr, V, H) — [μ; b] safety-corridor channels
        """
        depth_feature = self.image_backbone(depth)
        input_tensor = torch.cat((obs, depth_feature), 1)
        output = self.yopo_head(input_tensor)                                     # (B, 15+2nr, V, H)

        nr = self.radius_num
        endstate = torch.tanh(output[:, :self.params_per_bin])                    # (B, 14, V, H)
        score = torch.nn.functional.softplus(output[:, self.params_per_bin])      # (B, V, H)
        rl = output[:, self.params_per_bin + 1:]                                  # (B, 2nr, V, H)
        mu = torch.sigmoid(rl[:, :nr])
        b = self.radius_b_min + (self.radius_b_max - self.radius_b_min) * torch.sigmoid(rl[:, nr:])
        radius = torch.cat([mu, b], dim=1)                                        # (B, 2nr, V, H)
        return endstate, score, radius

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to (inner_pos, tail_pva, durations) in body frame.
            obs: current state in the body frame.
            return: inner_pos_b [B, V*H, 3], tail_pva_b [B, V*H, 3, 3], durations [B, V*H, 2],
                    score [B, V, H], inner_radio_offset [B, V*H] (raw Δr, 0 ⇒ centred),
                    radius [B, N, 2·nr] ([μ; b], grid-flattened like N, image order)
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred, radius_pred = self.forward(depth, obs)
        inner_pos_b, tail_pva_b, durations, inner_radio_offset = self.state_transform.pred_to_traj_params(endstate_pred)
        radius_flat = radius_pred.permute(0, 2, 3, 1).reshape(radius_pred.shape[0], -1, 2 * self.radius_num)   # (B, N, 2nr)
        return inner_pos_b, tail_pva_b, durations, score_pred, inner_radio_offset, radius_flat

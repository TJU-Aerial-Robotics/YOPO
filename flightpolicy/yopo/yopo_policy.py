"""
YOPO Network
forward, prediction, pre-processing, post-processing
"""

import torch as th
from torch import nn
import numpy as np
from typing import Any, Dict, List, Type
from flightpolicy.yopo.yopo_network import YopoBackbone, CostAndGradLayer


class YopoPolicy(nn.Module):

    def __init__(
            self,
            observation_dim,
            action_dim,  # x_pva, y_pva, z_pva, score
            hidden_state,
            lattice_space,
            lattice_primitive,
            lr_schedule=None,
            train_env=None,
            net_arch=None,
            activation_fn=nn.ReLU,
            normalize_images=True,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,
            device=None
    ):
        super(YopoPolicy, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lattice_space = lattice_space
        self.hidden_state = hidden_state
        self.lattice_primitive = lattice_primitive
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images
        self.yaw_diff = lattice_primitive.yaw_diff
        self.pitch_diff = lattice_primitive.pitch_diff
        self.train_env = train_env
        self.device = device

        self._build(lr_schedule)

    def _build(self, lr_schedule=None) -> None:
        # output state dim = action dim + score
        output_dim = (self.action_dim + 1) * self.lattice_space.vel_num * self.lattice_space.radio_num
        # input state dim = hidden_state + vel + acc + goal
        input_dim = self.hidden_state + 9
        self.image_backbone = YopoBackbone(self.hidden_state,
                                           self.lattice_space.horizon_num * self.lattice_space.vertical_num)
        self.state_backbone = nn.Sequential()
        self.yopo_header = self.create_header(input_dim, output_dim, self.net_arch, self.activation_fn, True)
        self.grad_layer = CostAndGradLayer.apply
        # Setup optimizer with initial learning rate
        learning_rate = lr_schedule(1) if lr_schedule is not None else 1e-3
        self.optimizer = self.optimizer_class(self.parameters(), lr=learning_rate)

    # TenserRT Transfer
    def forward(self, depth: th.Tensor, obs: th.Tensor) -> th.Tensor:
        """
            forward propagation of neural network, only used for TensorRT conversion.
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = th.cat((obs_feature, depth_feature), 1)
        output = self.yopo_header(input_tensor)
        # [batch, endstate+score, lattice_row, lattice_col]
        return output

    # Training Policy
    def inference(self, depth: th.Tensor, obs: th.Tensor) -> th.Tensor:
        """
            For network training:
            (1) predicted the endstate(end_state) and score
            (2) record the gradients and costs of prediction
        """
        depth_feature = self.image_backbone(depth)
        obs_feature = self.state_backbone(obs)
        input_tensor = th.cat((obs_feature, depth_feature), 1)
        output = self.yopo_header(input_tensor)

        # [batch, endstate+score, lattice_num]
        batch_size = obs.shape[0]
        output = output.view(batch_size, 10, self.lattice_space.horizon_num * self.lattice_space.vertical_num)
        # output.register_hook(self.print_grad)
        endstate_pred = output[:, 0:9, :]
        score_pred = output[:, 9, :]

        endstate_score_predictions = th.zeros_like(output).to(self.device)
        cost_labels = th.zeros((batch_size, self.lattice_space.horizon_num * self.lattice_space.vertical_num)).to(self.device)
        for i in range(0, self.lattice_space.horizon_num * self.lattice_space.vertical_num):
            id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - i
            ids = id * np.ones((batch_size, 1))
            endstate = self.pred_to_endstate(endstate_pred[:, :, i], id)
            # endstate.register_hook(self.print_grad)
            cost_label = self.grad_layer(endstate, self.train_env, ids)
            endstate_score_predictions[:, 0:9, i] = endstate
            endstate_score_predictions[:, 9, i] = score_pred[:, i]
            cost_labels[:, i] = cost_label.squeeze()

        return endstate_score_predictions, cost_labels

    # Testing Policy
    def predict(self, depth: th.Tensor, obs: th.Tensor, return_all_preds=False) -> th.Tensor:
        """
            For network testing:
            (1) predicted the endstate(end_state) and score
        """
        with th.no_grad():
            depth_feature = self.image_backbone(depth)
            obs_feature = self.state_backbone(obs.float())
            input_tensor = th.cat((obs_feature, depth_feature), 1)
            output = self.yopo_header(input_tensor)
            batch_size = obs.shape[0]
            output = output.view(batch_size, 10, self.lattice_space.horizon_num * self.lattice_space.vertical_num)
            endstate_pred = output[:, 0:9, :]
            score_pred = output[:, 9, :]

            if not return_all_preds:
                endstate_prediction = th.zeros(batch_size, self.action_dim)
                score_prediction = th.zeros(batch_size, 1)
                for i in range(0, batch_size):
                    action_id = th.argmin(score_pred[i]).item()
                    lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - action_id
                    endstate_prediction[i] = self.pred_to_endstate(th.unsqueeze(endstate_pred[i, :, action_id], 0), lattice_id)
                    score_prediction[i] = score_pred[i, action_id]
            else:
                endstate_prediction = th.zeros_like(endstate_pred)
                score_prediction = score_pred
                for i in range(0, self.lattice_space.horizon_num * self.lattice_space.vertical_num):
                    lattice_id = self.lattice_space.horizon_num * self.lattice_space.vertical_num - 1 - i
                    endstate = self.pred_to_endstate(endstate_pred[:, :, i], lattice_id)
                    endstate_prediction[:, :, i] = endstate

        return endstate_prediction, score_prediction

    def pred_to_endstate(self, endstate_pred: th.Tensor, id: int):
        """
            Transform the predicted state to the body frame.
        """
        delta_yaw = endstate_pred[:, 0] * self.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.pitch_diff
        radio = endstate_pred[:, 2] * self.lattice_space.radio_range + self.lattice_space.radio_range
        yaw, pitch = self.lattice_primitive.getAngleLattice(id)
        endstate_x = th.cos(pitch + delta_pitch) * th.cos(yaw + delta_yaw) * radio
        endstate_y = th.cos(pitch + delta_pitch) * th.sin(yaw + delta_yaw) * radio
        endstate_z = th.sin(pitch + delta_pitch) * radio
        endstate_p = th.stack((endstate_x, endstate_y, endstate_z), dim=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_space.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_space.acc_max
        Rbp = self.lattice_primitive.getRotation(id)
        endstate_vb = th.matmul(th.tensor(Rbp).to(self.device), endstate_vp.t()).t()
        endstate_ab = th.matmul(th.tensor(Rbp).to(self.device), endstate_ap.t()).t()
        endstate = th.cat((endstate_p, endstate_vb, endstate_ab), dim=1)
        endstate[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]] = endstate[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]
        return endstate

    def create_header(self,
                      input_dim: int,
                      output_dim: int,
                      net_arch: List[int],
                      activation_fn: Type[nn.Module] = nn.ReLU,
                      squash_output: bool = False,
                      ) -> nn.Sequential:

        if len(net_arch) > 0:
            modules = [nn.Conv2d(in_channels=input_dim, out_channels=net_arch[0], kernel_size=1, stride=1, padding=0),
                       activation_fn()]
        else:
            modules = []

        for idx in range(len(net_arch) - 1):
            modules.append(nn.Conv2d(in_channels=net_arch[idx], out_channels=net_arch[idx + 1], kernel_size=1, stride=1,
                                     padding=0))
            modules.append(activation_fn())

        if output_dim > 0:
            last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
            modules.append(nn.Conv2d(in_channels=last_layer_dim, out_channels=output_dim, kernel_size=1, stride=1,
                                     padding=0))
        if squash_output:
            modules.append(nn.Tanh())
        return nn.Sequential(*modules)

    def get_constructor_parameters(self) -> Dict[str, Any]:
        data = {"net_arch": self.net_arch,
                "hidden_state": self.hidden_state,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "activation_fn": self.activation_fn,
                "lattice_space": self.lattice_space,
                "lattice_primitive": self.lattice_primitive
                }
        return data

    def print_grad(ctx, grad):
        print("grad of hook: ", grad)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

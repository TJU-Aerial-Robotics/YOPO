# The backbone and the custom gradient layer.
import time
import torch as th
import torch.nn
import numpy as np
from torchvision.models import mobilenet_v3_small
from flightpolicy.yopo.resnet import resnet18
from torch.autograd import Function


# 18ms, Fast and effective.
class ResNet18(torch.nn.Module):
    def __init__(self, output_dim: int, primitive_shape: int):
        super(ResNet18, self).__init__()
        self.cnn = resnet18(pretrained=False)
        self.cnn.conv1 = th.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if (primitive_shape != 1):
            self.cnn.avgpool = th.nn.Sequential()
        self.cnn.fc = th.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.features_dim = output_dim

    def forward(self, depth: th.Tensor) -> th.Tensor:
        return self.cnn(depth)


# 20ms, Performs worse than ResNet and is slower than ResNet-18.
class MobileNet(th.nn.Module):
    def __init__(self, output_dim: int):
        super(MobileNet, self).__init__()
        self.cnn = mobilenet_v3_small(pretrained=False)
        self.cnn.features[0][0] = th.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn.classifier = th.nn.Linear(576, output_dim)
        self.features_dim = output_dim

    def forward(self, depth: th.Tensor) -> th.Tensor:
        return self.cnn(depth)


def YopoBackbone(output_dim, primitive_shape):
    return ResNet18(output_dim, primitive_shape)


class CostAndGradLayer(Function):

    @staticmethod
    def forward(ctx, input_dp, train_env, primitive_id):
        # print("input ", input_dp.shape)
        device = input_dp.device
        cost, grad = train_env.getCostAndGradient(input_dp, primitive_id)
        grad = np.minimum(grad, 1.0)  # Gradient clipping: Prevent excessively large values.
        cost = torch.tensor(cost).to(device)
        grad = torch.tensor(grad).to(device)
        ctx.save_for_backward(grad)
        cost.requires_grad = True
        return cost

    @staticmethod
    def backward(ctx, cost_grad_input):
        grad, = ctx.saved_tensors
        return_grad = th.bmm(grad.unsqueeze(-1), cost_grad_input.unsqueeze(-1)).squeeze(dim=2)
        # print("grad ", return_grad.shape)
        # print("grad: ", return_grad)
        return return_grad, None, None


if __name__ == '__main__':
    net = YopoBackbone(64, 3)
    input_ = torch.zeros((1, 1, 96, 96))
    start = time.time()
    output = net(input_)
    print(time.time() - start)

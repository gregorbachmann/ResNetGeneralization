import torch
import torch.nn as nn
import numpy as np


class ResNet(nn.Module):

    def __init__(self, input_dim, inter_dim, output_dim, depth, theta):
        super(ResNet, self).__init__()
        assert output_dim % 2 == 0
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        self.depth = depth
        self.theta = theta
        self.layers = nn.ModuleList()

        # Only until depth - 1 since last layer is fixed
        for i in range(self.depth - 1):
            if i == 0:
                layer = nn.Linear(in_features=input_dim, out_features=inter_dim, bias=False)
            elif i == self.depth - 2:
                layer = nn.Linear(in_features=inter_dim, out_features=output_dim, bias=False)
            else:
                layer = nn.Linear(in_features=inter_dim, out_features=inter_dim, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(size=layer.weight.shape) * np.sqrt(2 / layer.weight.shape[0]))
            self.layers.append(layer)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        with torch.no_grad():
            x = nn.functional.normalize(x, p=2, dim=1)
        v1 = [1.0 for _ in range(self.output_dim // 2)]
        v2 = [-1.0 for _ in range(self.output_dim // 2)]
        v = torch.tensor(v1 + v2)
        out = x
        for i, layer in enumerate(self.layers):
            if i == 0 or i == self.depth - 2:
                out = torch.relu(layer(out))
            else:
                out = torch.relu(layer(out)) + self.theta * out

        out = torch.matmul(out, v)

        return out

    def weight_norms(self, writer, it):
        for i, layer in enumerate(self.layers):
            norm = torch.norm(layer.weight).detach().numpy()
            writer.add_scalar('Norm at Layer ' + str(i), norm, it)

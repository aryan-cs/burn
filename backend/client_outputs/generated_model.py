import torch
import torch.nn as nn


class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0_node_2 = nn.Flatten()
        self.layer_1_node_3_linear = nn.Linear(784, 128)
        self.layer_2_node_3_act = nn.ReLU()
        self.layer_3_node_4_linear = nn.Linear(128, 10)
        self.layers = nn.ModuleList([self.layer_0_node_2, self.layer_1_node_3_linear, self.layer_2_node_3_act, self.layer_3_node_4_linear])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

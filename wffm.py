import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from torch.nn import init
class WFFM(nn.Module):
    def __init__(self, in_channels, num_encoding_strategies):
        super(WFFM, self).__init__()
        self.in_channels = in_channels
        self.num_encoding_strategies = num_encoding_strategies
        # Define the MLP: mapping from in_channels to num_encoding_strategies * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, num_encoding_strategies * in_channels)
        )
    def forward(self,features_list):
        # Step 1: Integrate features from x0 to x3
        # Step 1: Integrate features from x0 to x5
        x0, x1, x2, x3, x4, x5 = features_list[:6]  # Take the first six features
        x0 = x0.view(x0.size(0), -1, 14, 14)
        x1 = x1.view(x1.size(0), -1, 14, 14)
        x2 = x2.view(x2.size(0), -1, 14, 14)
        x3 = x3.view(x3.size(0), -1, 14, 14)
        x4 = x4.view(x4.size(0), -1, 14, 14)
        x5 = x5.view(x5.size(0), -1, 14, 14)
        # Integrate the features
        F = x0 + x1 + x2 + x3 + x4 + x5  # Sum the features
        # Step 2: Global Average Pooling
        g = F.mean(dim=(2, 3))  # Global Pooling over H and W -> shape: [batch_size, C]
        h = self.mlp(g)  # h shape: [batch_size, C * num_encoding_strategies]
        h = h.view(-1, self.num_encoding_strategies,
                   self.in_channels)  # Reshape -> shape: [batch_size, num_encoding_strategies, C]
        p = torch.softmax(h, dim=1)  # Apply softmax along the num_encoding_strategies dimension
        # Step 5: Weighted sum to obtain V
        x0_f, x1_f, x2_f, x3_f, x4_f, x5_f = torch.split(p, 1, dim=1)  # Split p -> each shape: [batch_size, 1, C]
        x0_p, x1_p, x2_p, x3_p, x4_p, x5_p = (
            x0_f.squeeze(1), x1_f.squeeze(1), x2_f.squeeze(1),
            x3_f.squeeze(1), x4_f.squeeze(1), x5_f.squeeze(1)
        )  # each shape: [batch_size, C]
        V = (x0_p[:, :, None, None] * x0 +
             x1_p[:, :, None, None] * x1 +
             x2_p[:, :, None, None] * x2 +
             x3_p[:, :, None, None] * x3 +
             x4_p[:, :, None, None] * x4 +
             x5_p[:, :, None, None] * x5)  # Sum up to x5
        return V

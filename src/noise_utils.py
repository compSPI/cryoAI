import torch
from torch import nn
import numpy as np


class AWGNGenerator(nn.Module):
    def __init__(self, use_noise=True, snr=0.1, power_signal=1):
        super(AWGNGenerator, self).__init__()
        if use_noise:
            self.sigma = power_signal / snr
        else:
            self.sigma = 0.

    def forward(self, proj):
        proj += self.sigma*torch.randn_like(proj)
        return proj
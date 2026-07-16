"""
TASU Projector Implementation
Copied from source_repo/Multitask/model/projector.py
EncoderProjectorLinearSiLU: LayerNorm -> Linear -> SiLU -> Linear
"""
import torch
import torch.nn as nn
import math


class EncoderProjectorLinearSiLU(nn.Module):
    """
    1. First perform LayerNorm to balance the mean and variance
    2. Replace ReLU with SiLU (with negative leakage)
    3. Bottleneck
    """
    def __init__(self, config, bottleneck=2048):
        super().__init__()
        in_dim = config.encoder_dim
        out_dim = config.llm_dim
        self.norm = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, bottleneck, bias=True),
            nn.SiLU(),
            nn.Linear(bottleneck, out_dim, bias=True),
        )
        nn.init.kaiming_uniform_(self.ffn[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.ffn[2].bias)
        self.k = 1

    def forward(self, x):
        x = self.norm(x)
        return self.ffn(x)

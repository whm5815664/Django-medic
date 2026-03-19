import torch
import torch.nn as nn
from src.utils.spectral_ops import compute_local_spectral_entropy

class SpectralBiasGenerator(nn.Module):
    """
    将局部谱熵转换为注意力偏置 (Attention Bias)。
    逻辑：高熵 (噪声) -> 大的负偏置 (抑制); 低熵 (结构) -> 接近 0 (保留)。
    """
    def __init__(self, window_size=3, init_scale=5.0, learnable=True):
        super().__init__()
        self.window_size = window_size
        
        if learnable:
            # 可学习的缩放因子 lambda
            self.scale_param = nn.Parameter(torch.ones(1) * init_scale)
        else:
            self.register_buffer('scale_param', torch.ones(1) * init_scale)

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
        Returns:
            bias: Attention bias map [B, 1, H, W]
        """
        # 1. 计算归一化谱熵 [0, 1]
        entropy_map = compute_local_spectral_entropy(x, self.window_size)
        
        # 2. 映射为负偏置: Bias = -lambda * Entropy
        # 高熵区域将得到如 -5.0 的值，Softmax 后趋近于 0
        bias = -self.scale_param * entropy_map
        
        return bias
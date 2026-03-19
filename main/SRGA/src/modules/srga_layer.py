import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.spectral_bias_gen import SpectralBiasGenerator

class SRGAAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=3, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 集成偏置生成器
        self.bias_gen = SpectralBiasGenerator(window_size=window_size)

    def forward(self, x, spatial_shape):
        """
        Args:
            x: Tensor [B, N, D] where N = H*W
            spatial_shape: Tuple (H, W)
        Returns:
            out: Tensor [B, N, D]
        """
        B, N, D = x.shape
        H, W = spatial_shape
        
        # 1. 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, Heads, N, HeadDim]
        
        # 2. 计算原始注意力分数
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # [B, Heads, N, N]
        
        # 3. 【核心】生成并注入谱秩偏置
        # 需要将序列格式 x 还原为空间格式 [B, C, H, W] 以计算局部熵
        x_spatial = x.transpose(1, 2).reshape(B, D, H, W)
        
        # 生成偏置 [B, 1, H, W]
        bias_map = self.bias_gen(x_spatial)
        
        # 将偏置展平为 [B, 1, 1, N] 以便广播到 Attention 矩阵
        # 注意：这里的偏置是加在 Key 的维度上的 (即最后一个维度 N)
        bias_flat = bias_map.reshape(B, 1, 1, H * W)
        
        # 应用偏置
        guided_scores = attn_scores + bias_flat
        
        # 4. Softmax & Dropout
        attn_probs = F.softmax(guided_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 5. Weighted Sum
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out

class SRGABlock(nn.Module):
    """标准的 Transformer Block，但使用 SRGAAttention"""
    def __init__(self, dim, num_heads, window_size=3, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRGAAttention(dim, num_heads, window_size, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, spatial_shape):
        x = x + self.attn(self.norm1(x), spatial_shape)
        x = x + self.mlp(self.norm2(x))
        return x
import torch
import math
import torch.nn.functional as F

def compute_local_spectral_entropy(x, window_size=3):
    """
    计算特征图局部的谱熵。
    Args:
        x: Tensor [B, C, H, W]
        window_size: int, 局部窗口大小 (e.g., 3)
    Returns:
        entropy_map: Tensor [B, 1, H, W], 归一化后的熵值 (0~1)
    """
    B, C, H, W = x.shape
    kh, kw = window_size, window_size
    pad = (kh - 1) // 2
    
    # 1. 局部窗口展开 (Unfold)
    # Padding to keep spatial dimensions
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    
    # Unfold: [B, C*kh*kw, L] where L = H*W
    patches = F.unfold(x_padded, kernel_size=(kh, kw))
    
    # Reshape to [B, C, kh*kw, H, W] then permute to [B, H, W, K, C]
    patches = patches.view(B, C, kh * kw, H, W)
    patches = patches.permute(0, 3, 4, 2, 1).contiguous() # [B, H, W, K, C]
    patches = patches.view(-1, kh * kw, C) # [B*H*W, K, C]
    
    # 2. 中心化 (Centering)
    patches = patches - patches.mean(dim=1, keepdim=True)
    
    # 3. 计算协方差矩阵 (Covariance)
    # We compute KxK covariance matrix since K (9) << C (channels) usually.
    # Cov = (1/(C-1)) * P * P^T
    cov = torch.matmul(patches, patches.transpose(-2, -1)) / (C - 1) # [B*H*W, K, K]
    
    # 4. 特征值分解 (Eigen Decomposition)
    # eigvalsh is for symmetric matrices, returns real eigenvalues
    eigvals = torch.linalg.eigvalsh(cov) # [B*H*W, K]
    
    # Numerical stability clamp
    eigvals = torch.clamp(eigvals, min=1e-6)
    
    # 5. 计算归一化谱熵 (Normalized Spectral Entropy)
    sum_eig = eigvals.sum(dim=-1, keepdim=True)
    p = eigvals / (sum_eig + 1e-6) # Probability distribution
    
    entropy = -torch.sum(p * torch.log(p + 1e-6), dim=-1) # [B*H*W]
    
    # Normalize to [0, 1] by max entropy log(K)
    max_entropy = math.log(kh * kw)
    norm_entropy = entropy / max_entropy
    
    # Reshape back to [B, 1, H, W]
    return norm_entropy.view(B, 1, H, W)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """简单的 CNN 编码器用于图像，保持空间特征以便后续 Transformer 处理"""
    def __init__(self, embed_dim):  # ✅ 修改参数名为 embed_dim
        super().__init__()
        # 基本卷积骨干，用于提取空间特征
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # downsample
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 不再使用 AdaptiveAvgPool2d，这里保留空间维度
        )
        # 将通道维度投影到期望的 embed_dim
        self.channel_proj = nn.Conv2d(128, embed_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        x = self.backbone(x)            # [B, 128, H_i, W_i]
        x = self.channel_proj(x)        # [B, embed_dim, H_i, W_i]
        return x

class AudioEncoder(nn.Module):
    """简单的 CNN 编码器用于音频 Mel 谱图，保留频谱的时频结构"""
    def __init__(self, embed_dim, in_channels=1): # ✅ 修改参数名为 embed_dim
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # 保留空间结构
        )
        self.channel_proj = nn.Conv2d(64, embed_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        x = self.backbone(x)            # [B, 64, F_out, T_out]
        x = self.channel_proj(x)        # [B, embed_dim, F_out, T_out]
        return x

class TabularEncoder(nn.Module):
    """MLP 编码器用于表格数据"""
    def __init__(self, input_dim, embed_dim): # ✅ 修改参数名为 embed_dim
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dtype == torch.double:
            x = x.float()
        elif x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        return self.net(x)
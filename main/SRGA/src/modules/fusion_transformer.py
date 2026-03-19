import torch
import torch.nn as nn
from src.modules.encoders import ImageEncoder, AudioEncoder, TabularEncoder
from src.modules.srga_layer import SRGABlock

class FusionTransformer(nn.Module):
    def __init__(self, 
                 img_embed_dim=128, 
                 audio_embed_dim=128, 
                 tabular_dim=3, 
                 fusion_dim=256,
                 num_heads=4, 
                 depth=2, 
                 window_size=3):
        super().__init__()
        
        # 1. 单模态编码器
        self.img_encoder = ImageEncoder(embed_dim=img_embed_dim)
        self.audio_encoder = AudioEncoder(embed_dim=audio_embed_dim)
        self.tab_encoder = TabularEncoder(input_dim=tabular_dim, embed_dim=fusion_dim)
        
        # 2. 投影层 (统一维度到 fusion_dim)
        self.img_proj = nn.Conv2d(img_embed_dim, fusion_dim, 1)
        self.audio_proj = nn.Conv2d(audio_embed_dim, fusion_dim, 1)
        
        # 3. SRGA Transformer Blocks
        # 注意：这里我们主要对图像和音频的空间序列进行自注意力融合
        # 表格数据作为全局 Context 或通过 Cross-Attention 引入，此处简化为拼接后处理
        self.blocks = nn.ModuleList([
            SRGABlock(dim=fusion_dim, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(fusion_dim)
        
        # 全局池化与分类准备
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, img, audio, tabular):
        """
        Args:
            img: [B, 3, H, W]
            audio: [B, 1, F, T]
            tabular: [B, tabular_dim]
        """
        B = img.shape[0]
        
        # 1. 编码
        img_feat = self.img_encoder(img)      # [B, D, H_i, W_i]
        audio_feat = self.audio_encoder(audio)# [B, D, F_a, T_a]
        tab_feat = self.tab_encoder(tabular)  # [B, fusion_dim]
        
        # 2. 投影与展平
        img_feat = self.img_proj(img_feat)    # [B, fusion_dim, H_i, W_i]
        audio_feat = self.audio_proj(audio_feat)
        
        # 展平为 Sequence: [B, N, D]
        B, D, H, W = img_feat.shape
        img_seq = img_feat.permute(0, 2, 3, 1).reshape(B, H*W, D)
        
        B, D, F, T = audio_feat.shape
        audio_seq = audio_feat.permute(0, 2, 3, 1).reshape(B, F*T, D)
        
        # 3. 融合策略：简单拼接序列 (Concatenation)
        # 总序列长度 = N_img + N_audio
        #fused_seq = torch.cat([img_seq, audio_seq], dim=1) # [B, N_total, D]
        
        # 记录图像部分的形状用于 SRGA 计算 (假设音频部分也复用图像的偏置逻辑或单独处理)
        # 为了简化演示，我们对整个拼接序列应用 SRGA，但在计算 Bias 时需小心空间结构。
        # *高级技巧*: 实际应用中，通常分别对 Img 和 Aud 做 SRGA，然后再 Cross-Attention。
        # 此处演示：仅对图像部分应用 SRGA，音频部分作为辅助，或者假设它们被重塑为一个大的虚拟网格。
        # 为了代码运行不报错且体现核心逻辑，我们仅对图像部分块应用 SRGA，音频部分做标准处理，
        # 或者更简单地：将整个序列视为一个长序列，但 SRGA 需要空间信息。
        
        # 【修正策略】：分别处理后再融合，或者只把图像作为主要空间关注对象
        # 这里演示：分别通过 SRGA 块处理图像序列，音频序列通过普通 MLP 或单独块，最后相加。
        # 但为了展示 `SRGABlock` 的威力，我们假设主要关注图像的空间噪声抑制。
        
        # 方案：只让图像序列通过 SRGA 块，音频序列通过简单的线性变换，然后 Concat 再过一个普通 Transformer?
        # 不，让我们坚持架构：将 Image 和 Audio 视为两个独立的 Token 流，分别通过 SRGA (如果 Audio 也有空间结构)
        # 由于 Audio 也是 2D (Freq, Time)，我们可以给它也赋予空间形状。
        
        # 处理 Image Stream with SRGA
        for block in self.blocks:
            img_seq = block(img_seq, spatial_shape=(H, W))
            
        # 处理 Audio Stream (此处简化，也可同样应用 SRGA 如果音频也是 2D 拓扑)
        # 为了演示多样性，假设音频也经过同样的 SRGA (需定义其形状)
        for block in self.blocks: # 共享权重或独立权重均可，这里复用
             audio_seq = block(audio_seq, spatial_shape=(F, T))
        
        # 全局池化获取向量
        img_vec = self.global_pool(img_seq.transpose(1, 2)).squeeze(-1) # [B, D]
        audio_vec = self.global_pool(audio_seq.transpose(1, 2)).squeeze(-1) # [B, D]
        
        # 融合：Img + Audio + Tabular
        final_repr = img_vec + audio_vec + tab_feat
        
        return self.norm(final_repr)
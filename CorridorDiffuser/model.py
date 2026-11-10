import torch
import torch.nn as nn
from diffusers import UNet2DModel, UNet2DConditionModel


class IIABlock(nn.Module):
    """
    信息集成注意力模块 (Information Integration Attention Block)
    参考 TGRS 2025 论文实现，通过双方向 1D 卷积捕捉空间序列信息。

    核心步骤：
    1. 特征分离：将输入特征图沿高度和宽度方向解耦
    2. 序列提取：使用平均池化和最大池化提取两个方向的特征序列
    3. 1D卷积建模：用一维卷积捕捉序列依赖关系
    4. 权重生成与融合：生成注意力权重，通过残差连接增强原始特征
    """

    def __init__(self, channels: int, kernel_size: int = 7, reduction: int = 16):
        super().__init__()
        self.conv_h = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),  # 修正
        )
        self.conv_w = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),  # 修正
        )

        # 融合两个方向特征的1x1卷积，生成注意力权重
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 将权重归一化到0~1之间
        )

        # 最终的通道变换，保持特征维度一致
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            增强后的特征图 (B, C, H, W)
        """
        B, C, H, W = x.shape

        # ----- 步骤1：提取高度方向 (H) 的序列特征 -----
        # 在宽度维度上做平均池化，压缩W维度，得到 (B, C, H)
        h_avg = x.mean(dim=3, keepdim=False)  # (B, C, H)
        # 通过1D卷积建模高度方向的序列依赖
        h_feat = self.conv_h(h_avg)  # (B, C, H)
        # 将特征扩展回原始空间尺寸，用于后续融合
        h_feat = h_feat.unsqueeze(-1).expand(-1, -1, -1, W)  # (B, C, H, W)

        # ----- 步骤2：提取宽度方向 (W) 的序列特征 -----
        # 在高度维度上做平均池化，压缩H维度，得到 (B, C, W)
        w_avg = x.mean(dim=2, keepdim=False)  # (B, C, W)
        # 通过1D卷积建模宽度方向的序列依赖
        w_feat = self.conv_w(w_avg)  # (B, C, W)
        # 将特征扩展回原始空间尺寸
        w_feat = w_feat.unsqueeze(2).expand(-1, -1, H, -1)  # (B, C, H, W)

        # ----- 步骤3：融合双方向特征，生成注意力权重 -----
        # 拼接两个方向的特征
        combined = torch.cat([h_feat, w_feat], dim=1)  # (B, 2C, H, W)
        # 生成注意力权重
        attention_weight = self.fusion(combined)  # (B, C, H, W)

        # ----- 步骤4：特征增强与残差连接 -----
        # 将注意力权重应用到原始特征上，并进行残差连接[reference:4]
        out = x * attention_weight + x

        # 可选的最终投影，保持特征一致性
        out = self.proj(out)

        return out


class IIAEncoder(nn.Module):
    """
    基于IIA模块的条件编码器，用于替换原有的ViT编码器。
    将输入的地图张量编码为条件特征序列，供UNet的cross-attention使用。
    """

    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 256,
                 depth: int = 2,
                 img_size: int = 256,
                 pool_size: int = 16,
                 kernel_size: int = 7,
                 reduction: int = 16):
        """
        Args:
            in_channels: 输入通道数 (地图为3)
            embed_dim: 特征嵌入维度
            depth: IIA模块堆叠层数
            img_size: 输入图像尺寸
            pool_size: 最终池化后的空间尺寸，控制token数量
            kernel_size: 1D卷积核大小
            reduction: 通道压缩比例
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.pool_size = pool_size

        # 输入投影：将输入通道映射到embed_dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 堆叠IIA模块
        self.iia_blocks = nn.Sequential(*[
            IIABlock(embed_dim, kernel_size=kernel_size, reduction=reduction)
            for _ in range(depth)
        ])

        # 自适应池化，将空间尺寸统一到 pool_size x pool_size
        # 这样token数量固定为 pool_size^2，方便与UNet的cross-attention对接
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # 输出投影，保持维度一致
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入地图张量 (B, 3, H, W)
        Returns:
            条件特征序列 (B, num_tokens, embed_dim)
            num_tokens = pool_size * pool_size
        """
        # 输入投影
        x = self.input_proj(x)  # (B, embed_dim, H, W)

        # 通过IIA模块进行空间特征增强
        x = self.iia_blocks(x)  # (B, embed_dim, H, W)

        # 池化压缩空间尺寸，控制token数量
        x = self.pool(x)  # (B, embed_dim, pool_size, pool_size)

        # 展平为序列格式 (B, num_tokens, embed_dim)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # 输出投影
        x = self.output_proj(x)  # (B, num_tokens, embed_dim)

        return x


class ConditionEncoder(nn.Module):
    def __init__(self,
                 encoder_type="cnn",
                 img_size=256,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 cross_attention_dim=256,
                 # IIA专用参数
                 iia_kernel_size: int = 7,
                 iia_reduction: int = 16,
                 iia_pool_size: int = 16
                 ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "cnn":
            # CNN Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(in_chans, 64, kernel_size=7, stride=4, padding=3),  # (B,64,64,64)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16)),  # (B,64,16,16)
            )
            self.flatten = nn.Flatten(2)  # -> (B,64,256)
            self.proj = nn.Linear(64, cross_attention_dim)
        elif encoder_type == "vit":
            self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                         kernel_size=patch_size, stride=patch_size)
            num_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        elif encoder_type == "iia":
            self.iia_encoder = IIAEncoder(
                in_channels=in_chans,
                embed_dim=embed_dim,
                depth=depth,  # IIA模块堆叠层数，建议2-4层即可
                img_size=img_size,
                pool_size=iia_pool_size,
                kernel_size=iia_kernel_size,
                reduction=iia_reduction
            )
            # 如果embed_dim和cross_attention_dim不一致，添加投影层
            if embed_dim != cross_attention_dim:
                self.proj = nn.Linear(embed_dim, cross_attention_dim)
            else:
                self.proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported encoder_type {encoder_type}")

    def forward(self, x):
        if self.encoder_type == "cnn":
            # CNN pipeline
            x = self.encoder(x)  # (B,64,16,16)
            x = self.flatten(x)  # (B,64,256)
            x = x.transpose(1, 2)  # (B,256,64)
            x = self.proj(x)  # (B,256,cross_attention_dim)

        elif self.encoder_type == "vit":
            # ViT pipeline
            x = self.patch_embed(x)  # (B,embed_dim,H/ps,W/ps)
            x = x.flatten(2).transpose(1, 2)  # (B,num_patches,embed_dim)
            x = x + self.pos_embed
            x = self.transformer(x)  # (B,num_patches,embed_dim)

        elif self.encoder_type == "iia":
            x = self.iia_encoder(x)  # (B, num_tokens, embed_dim)
            x = self.proj(x)  # (B, num_tokens, cross_attention_dim)

        return x  # (B, num_tokens, cross_attention_dim or embed_dim)


class InputFusion(nn.Module):
    """
    Small fusion block that takes concatenated [y_t (1) , map (3)] (=> 4 channels)
    and projects to a desired number of input channels for UNet.
    Keeps spatial resolution.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SafetyCorridorUNet(nn.Module):
    """
    UNet-based diffusion model using diffusers.UNet2DModel.
    Conditioning done by concatenating map tensor channels to y_t before UNet (in_channels=4).
    The UNet includes self-attention blocks in deeper layers for global context.

    Forward:
      - inputs:
          y_t: (B,1,H,W)  noisy target
          t:   (B,)       timesteps (long tensor)
          map_tensor: (B,3,H,W) binary map (0/1)
          alpha_cum_t: optional scalar or tensor for accurate y0 reconstruction
      - returns dict with:
          "eps_pred": predicted noise (B,1,H,W)
    """

    def __init__(
            self,
            sample_size: int = 256,
            in_channels: int = 4,  # 1 (y_t) + 3 (map)
            base_channels: int = 64,
            cross_dim: int = 256,
            use_input_fusion: bool = True,
            encoder_type: str = "vit",  # 新增：选择编码器类型 "cnn" | "vit" | "iia"
            # IIA专用参数
            iia_depth: int = 3,
            iia_kernel_size: int = 7,
            iia_reduction: int = 16,
            iia_pool_size: int = 16,
            # ViT专用参数
            vit_depth: int = 6,
            vit_num_heads: int = 8,
            vit_patch_size: int = 16,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_input_fusion = use_input_fusion

        # ----- 条件编码器 -----
        if encoder_type == "iia":
            # 使用IIA编码器
            self.cond_encoder = ConditionEncoder(
                encoder_type="iia",
                img_size=sample_size,
                in_chans=3,
                embed_dim=cross_dim,
                depth=iia_depth,
                cross_attention_dim=cross_dim,
                iia_kernel_size=iia_kernel_size,
                iia_reduction=iia_reduction,
                iia_pool_size=iia_pool_size,
            )
        elif encoder_type == "vit":
            # 使用ViT编码器
            self.cond_encoder = ConditionEncoder(
                encoder_type="vit",
                img_size=sample_size,
                patch_size=vit_patch_size,
                in_chans=3,
                embed_dim=cross_dim,
                depth=vit_depth,
                num_heads=vit_num_heads,
                cross_attention_dim=cross_dim,
            )
        elif encoder_type == "cnn":
            # 使用CNN编码器
            self.cond_encoder = ConditionEncoder(
                encoder_type="cnn",
                in_chans=3,
                cross_attention_dim=cross_dim,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        if use_input_fusion:
            self.fusion = InputFusion(in_channels=in_channels, out_channels=base_channels)
            unet_in_ch = base_channels
        else:
            self.fusion = None
            unet_in_ch = in_channels

        # Build UNet2DModel from diffusers.
        # Use self-attention in deeper blocks to capture global context.
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=unet_in_ch,
            out_channels=1,  # predict noise (eps)
            cross_attention_dim=cross_dim,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=8,
        )

    def forward(self, y_t: torch.Tensor, t: torch.Tensor, map_tensor: torch.Tensor):
        """
        y_t: (B,1,H,W)  noisy speed map (range depends on dataset preprocessing, e.g. [-1,1])
        t:   (B,) long tensor of timesteps
        map_tensor: (B,3,H,W) binary map channels (0/1)s
        """

        # concat along channel dim
        inp = torch.cat([y_t, map_tensor], dim=1)  # (B,4,H,W)

        if self.fusion is not None:
            inp = self.fusion(inp)  # (B, base_channels, H, W)

        cond_feat = self.cond_encoder(map_tensor)  # (B,64,16,16)

        out = self.unet(inp, t, cond_feat)
        eps_pred = out.sample  # (B,1,H,W)

        return eps_pred


#
if __name__ == "__main__":
    # simple smoke test to ensure shapes flow
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SafetyCorridorUNet(sample_size=256, use_input_fusion=True, encoder_type='iia').to(device)

    B = 8
    y_t = torch.randn(B, 1, 256, 256, device=device)  # noisy input (e.g. in [-1,1])
    t = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
    map_tensor = torch.randint(0, 2, (B, 3, 256, 256), device=device).float()

    out = model(y_t, t, map_tensor)
    print("eps_pred.shape:", out.shape)

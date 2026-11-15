"""
EndoViT (Endoscopic Vision Transformer) Backbone for Method 1 (EDN).
Based on: "EndoViT: pretraining vision transformers on a large collection of endoscopic images"
(Dominik Batić, Felix Holm, Ege Özsoy, Tobias Czempiel, Nassir Navab, 2024).
Repository: https://github.com/DominikBatic/EndoViT
Hugging Face: https://huggingface.co/egeozsoy/EndoViT

Includes:
- Vision Transformer (ViT-Base, patch_size=16, embed_dim=768, depth=12)
- MultiScaleFeatureAdapter: Converts flat ViT patch tokens from intermediate blocks
  into hierarchical 2D feature maps at 1/4, 1/8, 1/16, and 1/32 scales to seamlessly
  interface with the EDN Decoder and Attention Skip Connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HUGGINGFACE_REPO_ID = "egeozsoy/EndoViT"


class ViTPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Standard Vision Transformer Block with Pre-LayerNorm."""

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MultiScaleFeatureAdapter(nn.Module):
    """
    Multiscale Spatial Token Adapter for EndoViT.
    Projects 1D ViT token sequences at H/16, W/16 into hierarchical 2D pyramid representations:
      - Stage 1 (from block 3):  Transposed Conv 4x -> Scale 1/4 (e.g., 336 ch for EDN skip 1)
      - Stage 2 (from block 6):  Transposed Conv 2x -> Scale 1/8 (e.g., 704 ch for EDN skip 2)
      - Stage 3 (from block 9):  Conv 1x1           -> Scale 1/16 (e.g., 1552 ch for EDN skip 3)
      - Stage 4 (from block 12): Strided Conv 2x   -> Scale 1/32 (e.g., 2688 ch for EDN bottleneck)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        out_channels: dict = None,
    ):
        super().__init__()
        if out_channels is None:
            # Matches DPN-92 channel profile to allow drop-in compatibility with EDN
            out_channels = {
                "head": 64,
                "layer1": 240,
                "layer2": 512,
                "layer3": 1248,
                "layer4": 1920,
            }

        self.out_channels = out_channels

        # Stem / Head projection (from input image)
        self.head_proj = nn.Sequential(
            nn.Conv2d(3, out_channels["head"], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels["head"]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 1: 1/16 -> 1/4 (4x upsample)
        self.proj_l1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, out_channels["layer1"], kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels["layer1"]),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 1/16 -> 1/8 (2x upsample)
        self.proj_l2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, out_channels["layer2"], kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels["layer2"]),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 1/16 -> 1/16 (1x scale)
        self.proj_l3 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels["layer3"], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels["layer3"]),
            nn.ReLU(inplace=True),
        )

        # Stage 4: 1/16 -> 1/32 (2x downsample)
        self.proj_l4 = nn.Sequential(
            nn.Conv2d(embed_dim, out_channels["layer4"], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels["layer4"]),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        img: torch.Tensor,
        token_maps: dict,
        grid_size: tuple,
    ) -> dict:
        """
        token_maps contains token sequences (B, N, embed_dim) from blocks 3, 6, 9, 12.
        """
        Gh, Gw = grid_size
        B = img.shape[0]

        # Reshape tokens into 2D feature maps: (B, embed_dim, Gh, Gw)
        f_b3 = token_maps[3].permute(0, 2, 1).reshape(B, -1, Gh, Gw)
        f_b6 = token_maps[6].permute(0, 2, 1).reshape(B, -1, Gh, Gw)
        f_b9 = token_maps[9].permute(0, 2, 1).reshape(B, -1, Gh, Gw)
        f_b12 = token_maps[12].permute(0, 2, 1).reshape(B, -1, Gh, Gw)

        f_head = self.head_proj(img)
        f_l1 = self.proj_l1(f_b3)
        f_l2 = self.proj_l2(f_b6)
        f_l3 = self.proj_l3(f_b9)
        f_l4 = self.proj_l4(f_b12)

        return {
            "head": f_head,
            "layer1": f_l1,
            "layer2": f_l2,
            "layer3": f_l3,
            "layer4": f_l4,
        }


class EndoViTBackbone(nn.Module):
    """
    EndoViT Vision Transformer Backbone with Endoscopy Domain Pretraining.
    Configured for 256x256 image inputs with 16x16 patch embeddings.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        pretrained: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = ViTPatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Multiscale Adapter to feed the EDN decoder
        self.adapter = MultiScaleFeatureAdapter(embed_dim=embed_dim)

        self._init_weights()
        if pretrained:
            self.load_pretrained_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape
        Gh, Gw = H // self.patch_size, W // self.patch_size

        tokens = self.patch_embed(x)

        # Interpolate position embedding if input dimensions differ from default
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos_tokens = self.pos_embed.reshape(1, int(math.sqrt(self.pos_embed.shape[1])), -1, self.embed_dim)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(Gh, Gw), mode="bicubic", align_corners=False)
            pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            pos_embed = self.pos_embed

        tokens = self.pos_drop(tokens + pos_embed)

        saved_stages = {}
        for idx, block in enumerate(self.blocks, start=1):
            tokens = block(tokens)
            if idx in (3, 6, 9, 12):
                saved_stages[idx] = tokens

        features = self.adapter(img=x, token_maps=saved_stages, grid_size=(Gh, Gw))
        return features

    def load_pretrained_weights(self):
        """Loads weights from Hugging Face Hub (egeozsoy/EndoViT) or local cache."""
        try:
            from huggingface_hub import hf_hub_download
            print("[EndoViT] Attempting to download EndoViT pretrained weights from Hugging Face...")
            try:
                ckpt_path = hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename="pytorch_model.bin")
            except Exception:
                ckpt_path = hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename="EndoViT_Seg.pth")

            state_dict = torch.load(ckpt_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            model_dict = self.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched)
            self.load_state_dict(model_dict)
            print(f"[EndoViT] Successfully loaded {len(matched)} weights from {HUGGINGFACE_REPO_ID}.")
        except Exception as e:
            print(f"[EndoViT] Note: Could not fetch remote weights ({e}). Initialized with random weights.")

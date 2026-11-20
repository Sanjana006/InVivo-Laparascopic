"""EndoSmoke-Net: HDMGN + frozen EndoViT + conditional flow-matching U-Net."""
from __future__ import annotations
import math
from functools import partial
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PlainResidualBlock(nn.Module):
    """Residual block without flow-time conditioning; used by HDMGN."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class HDMGN(nn.Module):
    """Compact hierarchical smoke-density guidance network."""
    def __init__(self, in_channels: int = 3, width: int = 16):
        super().__init__()
        self.stem = ConvNormAct(in_channels, width)
        self.enc1 = nn.Sequential(ConvNormAct(width, width * 2, 2), PlainResidualBlock(width * 2, width * 2))
        self.enc2 = nn.Sequential(ConvNormAct(width * 2, width * 3, 2), PlainResidualBlock(width * 3, width * 3))
        self.enc3 = nn.Sequential(ConvNormAct(width * 3, width * 4, 2), PlainResidualBlock(width * 4, width * 4))
        self.dec2 = ConvNormAct(width * 4 + width * 3, width * 3)
        self.dec1 = ConvNormAct(width * 3 + width * 2, width * 2)
        self.dec0 = ConvNormAct(width * 2 + width, width)
        self.head = nn.Sequential(nn.Conv2d(width, 1, 1), nn.Sigmoid())

    def forward(self, image):
        s0 = self.stem(image)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        z = self.enc3(s2)
        z = F.interpolate(z, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec2(torch.cat([z, s2], dim=1))
        z = F.interpolate(z, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec1(torch.cat([z, s1], dim=1))
        z = F.interpolate(z, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        z = self.dec0(torch.cat([z, s0], dim=1))
        return self.head(z)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        phase = t[:, None] * freq[None, :]
        emb = torch.cat([phase.sin(), phase.cos()], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class TimeResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.time = nn.Linear(time_dim, out_ch * 2)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, time_embedding):
        h = self.act(self.norm1(self.conv1(x)))
        scale, shift = self.time(time_embedding).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.norm2(self.conv2(self.act(h)))
        return self.act(h + self.skip(x))


class FrozenEndoViT(nn.Module):
    """EndoViT ViT-Base/16 feature extractor with checkpoint-load diagnostics."""
    def __init__(self, checkpoint=None, repo_id="egeozsoy/EndoViT", model_filename="pytorch_model.bin",
                 freeze=True, input_size=224, min_loaded_fraction=0.95):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as exc:
            raise ImportError("Install timm: pip install timm") from exc
        self.freeze = freeze
        self.input_size = input_size
        self.encoder = VisionTransformer(
            img_size=input_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        if checkpoint:
            checkpoint_obj = torch.load(checkpoint, map_location="cpu")
        else:
            try:
                from huggingface_hub import hf_hub_download
                weight_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
                checkpoint_obj = torch.load(weight_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(
                    "Could not obtain EndoViT weights. Supply --endovit_checkpoint PATH. "
                    "Check the official EndoViT repository for its current checkpoint instructions."
                ) from exc

        state = checkpoint_obj.get("model", checkpoint_obj)
        if not isinstance(state, dict):
            raise TypeError("EndoViT checkpoint must contain a state dictionary or a 'model' state dictionary.")
        # Strip common distributed-training prefix without altering parameter names otherwise.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        incompatible = self.encoder.load_state_dict(state, strict=False)
        own_keys = set(self.encoder.state_dict())
        loaded_keys = own_keys.intersection(state.keys())
        loaded_fraction = len(loaded_keys) / max(len(own_keys), 1)
        missing_core = [k for k in incompatible.missing_keys if not k.startswith("head.")]
        print(
            f"[EndoViT] loaded parameter-key coverage: {loaded_fraction:.1%}; "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )
        if missing_core or loaded_fraction < min_loaded_fraction:
            examples = missing_core[:8] or incompatible.missing_keys[:8]
            raise RuntimeError(
                "EndoViT checkpoint appears incompatible or incompletely loaded. "
                f"Example missing keys: {examples}. Verify checkpoint format and model definition."
            )
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, image):
        # Input images are stored in [-1, 1]; EndoViT normalization expects [0, 1].
        x = F.interpolate(image, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x.clamp(-1, 1) + 1.0) / 2.0
        mean = x.new_tensor([0.3464, 0.2280, 0.2228])[None, :, None, None]
        std = x.new_tensor([0.2520, 0.2128, 0.2093])[None, :, None, None]
        x = (x - mean) / std
        with torch.set_grad_enabled(not self.freeze):
            tokens = self.encoder.forward_features(x)
        if tokens.ndim == 3:
            # Discard CLS token and reshape patch tokens to a spatial feature map.
            tokens = tokens[:, 1:, :]
            side = int(math.sqrt(tokens.shape[1]))
            if side * side != tokens.shape[1]:
                raise RuntimeError(f"Expected square ViT patch grid, got {tokens.shape[1]} patch tokens.")
            tokens = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        return tokens


class EndoSmokeCCFMUNet(nn.Module):
    """7-channel conditional flow U-Net; EndoViT features enter at the bottleneck."""
    def __init__(self, base=32, time_dim=128, endovit_checkpoint=None,
                 use_endovit=True, freeze_endovit=True):
        super().__init__()
        self.use_endovit = use_endovit
        self.time = TimeEmbedding(time_dim)
        self.endovit = FrozenEndoViT(endovit_checkpoint, freeze=freeze_endovit) if use_endovit else None
        if use_endovit:
            self.vit_projection = nn.Sequential(
                nn.Conv2d(768, base * 8, 1, bias=False),
                nn.GroupNorm(_groups(base * 8), base * 8), nn.SiLU(inplace=True)
            )
        self.stem = ConvNormAct(7, base)
        self.enc1 = TimeResidualBlock(base, base * 2, time_dim)
        self.enc2 = TimeResidualBlock(base * 2, base * 4, time_dim)
        self.enc3 = TimeResidualBlock(base * 4, base * 8, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid1 = TimeResidualBlock(base * 8, base * 8, time_dim)
        self.mid2 = TimeResidualBlock(base * 8, base * 8, time_dim)
        self.dec2 = TimeResidualBlock(base * 8 + base * 4, base * 4, time_dim)
        self.dec1 = TimeResidualBlock(base * 4 + base * 2, base * 2, time_dim)
        self.dec0 = TimeResidualBlock(base * 2 + base, base, time_dim)
        self.output = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x_t, smoky, density, t, vit_features=None):
        # Exactly seven channels: flow state (3) + smoky observation (3) + density (1).
        time_emb = self.time(t)
        h0 = self.stem(torch.cat([x_t, smoky, density], dim=1))
        h1 = self.enc1(self.pool(h0), time_emb)
        h2 = self.enc2(self.pool(h1), time_emb)
        h3 = self.enc3(self.pool(h2), time_emb)
        h = self.mid1(self.pool(h3), time_emb)
        if self.use_endovit:
            if vit_features is None:
                vit_features = self.endovit(smoky)
            v = self.vit_projection(vit_features)
            h = h + F.interpolate(v, size=h.shape[-2:], mode="bilinear", align_corners=False)
        h = self.mid2(h, time_emb)
        h = F.interpolate(h, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec2(torch.cat([h, h2], dim=1), time_emb)
        h = F.interpolate(h, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec1(torch.cat([h, h1], dim=1), time_emb)
        h = F.interpolate(h, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec0(torch.cat([h, h0], dim=1), time_emb)
        return self.output(h)


class EndoSmokeNet(nn.Module):
    def __init__(self, endovit_checkpoint=None, use_endovit=True, freeze_endovit=True, base=32):
        super().__init__()
        self.density_net = HDMGN()
        self.flow_net = EndoSmokeCCFMUNet(
            base=base, endovit_checkpoint=endovit_checkpoint,
            use_endovit=use_endovit, freeze_endovit=freeze_endovit
        )

    def forward(self, x_t, smoky, t):
        density = self.density_net(smoky)
        # Explicit feature extraction makes the conditioning path visible and testable.
        # It is one EndoViT pass per training forward, not one pass per ODE step.
        vit_features = self.flow_net.endovit(smoky) if self.flow_net.use_endovit else None
        velocity = self.flow_net(x_t, smoky, density, t, vit_features=vit_features)
        return velocity, density

    @torch.no_grad()
    def sample(self, smoky, steps=20, seed=None, solver="euler"):
        """Sample a clean image using Euler or classical fixed-step RK4 integration."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if solver not in {"euler", "rk4"}:
            raise ValueError("solver must be 'euler' or 'rk4'")
        self.eval()
        batch, _, height, width = smoky.shape
        generator = None
        if seed is not None:
            generator = torch.Generator(device=smoky.device).manual_seed(seed)
        x = torch.randn((batch, 3, height, width), device=smoky.device,
                        dtype=smoky.dtype, generator=generator)
        density = self.density_net(smoky)
        vit_features = self.flow_net.endovit(smoky) if self.flow_net.use_endovit else None
        dt = 1.0 / steps

        def velocity(state, time_value):
            t = torch.full((batch,), time_value, device=smoky.device, dtype=smoky.dtype)
            return self.flow_net(state, smoky, density, t, vit_features=vit_features)

        for index in range(steps):
            t0 = index * dt
            if solver == "euler":
                x = x + dt * velocity(x, t0)
            else:
                k1 = velocity(x, t0)
                k2 = velocity(x + 0.5 * dt * k1, t0 + 0.5 * dt)
                k3 = velocity(x + 0.5 * dt * k2, t0 + 0.5 * dt)
                k4 = velocity(x + dt * k3, t0 + dt)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x.clamp(-1, 1), density

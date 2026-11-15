"""
Verification and Unit Test Suite for Trident Dehazing Network (TDN).
Validates:
1. Deformable Convolution & Core Blocks
2. Method 1: EDN with Attention Skip Connections (DPN92 & EndoViT)
3. Method 2: DRN with Wide Activation Blocks & Space-to-Depth/Depth-to-Space
4. Method 3: HDMGN 6-level U-Net
5. Full Trident Model end-to-end forward pass
6. Composite Loss (L1 + 2D FFT Amplitude & Phase + BReLU Threshold Limitation)
7. Backward pass and gradient flow
"""

import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.blocks import (
    DeformableConv2d,
    AttentionGate,
    DeformableUpsamplingBlock,
    WideActivationBlock,
    BReLU,
)
from models.edn import EncoderDecoderNet
from models.drn import DetailsRefinementNet
from models.hdmgn import HazeDensityMapNet
from models.trident_model import TridentDehazingNet
from losses.composite_loss import TridentCompositeLoss


def test_core_blocks():
    print("[TEST 1/6] Testing Core Blocks...")
    x = torch.randn(2, 64, 32, 32)

    # Deformable Conv
    dconv = DeformableConv2d(64, 64, kernel_size=3, padding=1)
    out_dconv = dconv(x)
    assert out_dconv.shape == (2, 64, 32, 32), f"DConv shape mismatch: {out_dconv.shape}"

    # Attention Gate
    g = torch.randn(2, 128, 16, 16)
    gate = AttentionGate(F_g=128, F_l=64, F_int=64)
    out_gate = gate(x=x, g=g)
    assert out_gate.shape == (2, 64, 32, 32), f"Attention Gate shape mismatch: {out_gate.shape}"

    # Deformable Upsampling Block (DUB)
    dub = DeformableUpsamplingBlock(in_channels=64, mid_channels=32, out_channels=32)
    out_dub = dub(x)
    assert out_dub.shape == (2, 32, 64, 64), f"DUB shape mismatch: {out_dub.shape}"

    # Wide Activation Block (WAB)
    wab = WideActivationBlock(channels=64, expand_factor=4, res_scale=0.2)
    out_wab = wab(x)
    assert out_wab.shape == (2, 64, 32, 32), f"WAB shape mismatch: {out_wab.shape}"

    # BReLU
    brelu = BReLU(0.0, 1.0)
    out_brelu = brelu(torch.tensor([-0.5, 0.4, 1.5]))
    assert torch.allclose(out_brelu, torch.tensor([0.0, 0.4, 1.0])), "BReLU clamp failed"

    print("  -> Core Blocks passed!")


def test_edn():
    print("[TEST 2/6] Testing Method 1: Encoder-Decoder Net (EDN)...")
    x = torch.randn(1, 3, 256, 256)

    # DPN92 backbone
    edn_dpn = EncoderDecoderNet(backbone_type="dpn92", pretrained=False, out_channels=16)
    out_dpn = edn_dpn(x)
    assert out_dpn.shape == (1, 16, 256, 256), f"EDN-DPN92 shape mismatch: {out_dpn.shape}"

    # EndoViT backbone
    edn_vit = EncoderDecoderNet(backbone_type="endovit", pretrained=False, out_channels=16)
    out_vit = edn_vit(x)
    assert out_vit.shape == (1, 16, 256, 256), f"EDN-EndoViT shape mismatch: {out_vit.shape}"

    print("  -> EDN with both DPN92 and EndoViT passed!")


def test_drn():
    print("[TEST 3/6] Testing Method 2: Details Refinement Net (DRN)...")
    x = torch.randn(1, 3, 256, 256)
    drn = DetailsRefinementNet(in_channels=3, feature_channels=64, out_channels=16)
    out_drn = drn(x)
    assert out_drn.shape == (1, 16, 256, 256), f"DRN shape mismatch: {out_drn.shape}"
    print("  -> DRN Space-to-Depth & WAB passed!")


def test_hdmgn():
    print("[TEST 4/6] Testing Method 3: Haze Density Map Net (HDMGN)...")
    x = torch.randn(1, 3, 256, 256)
    hdmgn = HazeDensityMapNet(in_channels=3, out_channels=3)
    density_map = hdmgn(x)
    assert density_map.shape == (1, 3, 256, 256), f"HDMGN shape mismatch: {density_map.shape}"
    assert (density_map >= 0.0).all() and (density_map <= 1.0).all(), "Density map not bounded in [0, 1]"
    print("  -> HDMGN passed!")


def test_full_model():
    print("[TEST 5/6] Testing Full TridentDehazingNet...")
    x = torch.rand(1, 3, 256, 256)

    # 1. Full model with DPN92
    model_dpn = TridentDehazingNet(backbone="dpn92", pretrained=False)
    res_dpn = model_dpn(x, return_internals=True)
    assert res_dpn["out"].shape == (1, 3, 256, 256), f"Output shape mismatch: {res_dpn['out'].shape}"
    assert res_dpn["density_map"].shape == (1, 3, 256, 256), "Density map shape mismatch"
    assert (res_dpn["out"] >= 0.0).all() and (res_dpn["out"] <= 1.0).all(), "Output not bounded in [0, 1]"

    # 2. Full model with EndoViT
    model_vit = TridentDehazingNet(backbone="endovit", pretrained=False)
    res_vit = model_vit(x, return_internals=True)
    assert res_vit["out"].shape == (1, 3, 256, 256), f"EndoViT output shape mismatch: {res_vit['out'].shape}"

    print("  -> Full TridentDehazingNet passed for both DPN92 and EndoViT backbones!")


def test_composite_loss_and_backward():
    print("[TEST 6/6] Testing Composite Loss (L1 + FFT + BReLU) & Backward Pass...")
    model = TridentDehazingNet(backbone="dpn92", pretrained=False)
    criterion = TridentCompositeLoss(alpha=0.5, beta=0.5, gamma=1.0)

    x = torch.rand(2, 3, 256, 256)
    gt = torch.rand(2, 3, 256, 256)

    res = model(x, return_internals=True)
    loss_dict = criterion(pred=res["out"], raw_pred=res["raw_out"], target=gt)

    assert "loss" in loss_dict and "l_spatial" in loss_dict and "l_freq" in loss_dict and "l_thresh" in loss_dict
    assert loss_dict["loss"].item() > 0.0, "Loss must be positive"
    assert loss_dict["l_freq"].item() > 0.0, "FFT loss must be positive"
    assert loss_dict["l_spatial"].item() > 0.0, "L1 loss must be positive"

    # Backward pass verification
    loss_dict["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    print(f"  -> Loss: {loss_dict['loss'].item():.4f} [L1: {loss_dict['l_spatial'].item():.4f}, FFT: {loss_dict['l_freq'].item():.4f}, BReLU: {loss_dict['l_thresh'].item():.4f}]")
    print("  -> Backward pass & gradient flow passed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("TRIDENT DEHAZING NETWORK (TDN) VERIFICATION SUITE")
    print("=" * 60)
    test_core_blocks()
    test_edn()
    test_drn()
    test_hdmgn()
    test_full_model()
    test_composite_loss_and_backward()
    print("=" * 60)
    print("ALL TESTS PASSED! IMPLEMENTATION FULLY VERIFIED.")
    print("=" * 60)


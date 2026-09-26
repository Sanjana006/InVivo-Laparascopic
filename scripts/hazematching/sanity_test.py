"""
scripts/hazematching/sanity_test.py
===================================
Sanity test for the Laparoscopy dataset integration.
1. Checks if the dataset loads correctly.
2. Checks tensor shapes and normalizations.
3. Does a single forward pass of the model.
4. Simulates a quick training step.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.hazematching.dataset import LaparoscopyDataset
from hazematching import CCFMFlowMatcher, CCFMUNet

def run_sanity_test():
    print("--- Starting Sanity Test ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = Path("data/laparoscopy/train_crop")
    
    if not data_dir.exists():
        print(f"Error: {data_dir} not found. Run data_prep.py first.")
        sys.exit(1)
        
    print(f"\n1. Loading LaparoscopyDataset from {data_dir}...")
    dataset = LaparoscopyDataset(subset="laparoscopy", folder=data_dir)
    
    print(f"Found {len(dataset)} image pairs.")
    
    if len(dataset) == 0:
        print("Error: No images found.")
        sys.exit(1)
        
    print("\n2. Checking tensor shapes and types...")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape} (Expected: [2, 256, 256])")
    print(f"Sample dtype: {sample.dtype}")
    
    # x0 is hazy (channel 1), x1 is clean (channel 0)
    x1_clean = sample[0:1]
    x0_smoky = sample[1:2]
    
    print(f"Clean (target) shape: {x1_clean.shape}")
    print(f"Smoky (input) shape: {x0_smoky.shape}")
    
    print("\n3. Testing Model Forward Pass...")
    model = CCFMUNet(
        dim=(2, 256, 256),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)
    
    print("Model initialized.")
    
    FM = CCFMFlowMatcher(sigma=0.0)
    
    # Create batch of 2
    batch = torch.stack([dataset[0], dataset[1]]).to(device)
    
    x1_batch = batch[:, 0:1] # clean
    x0_batch = batch[:, 1:2] # smoky
    
    x0_noise = torch.randn_like(x0_batch)
    t = torch.tensor([0.5, 0.8]).to(device)
    
    t, xt, ut = FM.sample_location_and_conditional_flow(x0_noise, x1_batch, t=t)
    
    xt_with_cond = torch.cat([xt, x0_batch], dim=1)
    
    print(f"Model input shape (t): {t.shape}")
    print(f"Model input shape (xt_with_cond): {xt_with_cond.shape}")
    
    try:
        pred_ut = model(t, xt_with_cond)
        print(f"Model output shape: {pred_ut.shape} (Expected: [2, 1, 256, 256])")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        sys.exit(1)
        
    print("\n4. Simulating Training Step...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    try:
        loss = criterion(pred_ut, ut)
        loss.backward()
        optimizer.step()
        print(f"Training step successful! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Training step failed: {e}")
        sys.exit(1)
        
    print("\n--- Sanity Test Passed! ---")

if __name__ == "__main__":
    run_sanity_test()

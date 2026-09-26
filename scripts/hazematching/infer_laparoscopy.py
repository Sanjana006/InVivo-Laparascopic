"""
scripts/hazematching/infer_laparoscopy.py
=========================================
Inference script tailored for the laparoscopy dataset.
Based on external/HazeMatching/scripts/infer.py
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import typer

# Ensure src is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.hazematching.dataset import LaparoscopyDataset
from hazematching import CCFMUNet
from hazematching.datasets.data_norm import normalize, denormalize

# Import odeint reliably (resolving module vs function collision in hazematching package)
try:
    from hazematching.odeint.odeint import odeint
except ImportError:
    try:
        from hazematching.odeint import odeint
    except ImportError:
        from hazematching import odeint

while not callable(odeint) and hasattr(odeint, "odeint"):
    odeint = odeint.odeint

warnings.filterwarnings("ignore")

app = typer.Typer()
SUBSET_NAME = "laparoscopy"

@app.command()
def infer(
    checkpoint: Annotated[Path, typer.Option(help="Path to model .pth checkpoint.")],
    data_dir: Annotated[Path, typer.Option(help="Root data directory.")] = Path("data"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Where to write results.")] = Path("outputs/hazematching"),
    folders: Annotated[Optional[list[str]], typer.Option(help="Which split folders to run.")] = None,
    n_samples: Annotated[int, typer.Option(help="Number of stochastic samples per image.")] = 5, # Default to 5 for speed
    num_steps: Annotated[int, typer.Option(help="Number of ODE time steps.")] = 20,
    max_batch_size: Annotated[int, typer.Option(help="Max images per ODE batch.")] = 8,
):
    if folders is None:
        folders = ["test"]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subset_dir = data_dir / SUBSET_NAME
    
    # Model uses full 256x256 image directly (no patches for laparoscopy)
    image_size = 256

    print(f"Loading model from {checkpoint}...")
    model = CCFMUNet(
        dim=(2, image_size, image_size),
        num_channels=32,
        out_channels=1,
        num_res_blocks=1,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ts = torch.linspace(0.0, 1.0, num_steps).to(device)

    for folder in folders:
        folder_dir = subset_dir / folder / "smoky"
        if not folder_dir.exists():
            print(f"Warning: {folder_dir} does not exist, skipping.")
            continue
            
        image_files = sorted(f for f in os.listdir(folder_dir) if f.endswith(".png"))

        out = output_dir / f"{folder}_results"
        out.mkdir(parents=True, exist_ok=True)

        for image_file in tqdm(image_files, desc=f"{folder}", unit="image"):
            img_path = folder_dir / image_file
            
            # Load Smoky (Grayscale)
            img = Image.open(img_path).convert('L')
            raw = np.array(img, dtype=np.float32)
            
            # Add channel dimension: (1, 256, 256)
            raw = np.expand_dims(raw, axis=0)
            
            # Normalize Smoky (Channel 1 in STATS)
            smoky_norm = normalize(raw, SUBSET_NAME, channel=1, path=None)
            
            condition = torch.from_numpy(smoky_norm).to(device).unsqueeze(0) # (1, 1, 256, 256)

            samples = []

            for sample_idx in tqdm(range(n_samples), desc="samples", leave=False):
                # We can batch the ODE step if we expand the condition
                noise = torch.randn_like(condition)
                input_tensor = torch.cat([noise, condition], dim=1) # (1, 2, 256, 256)
                
                with torch.no_grad():
                    traj, _ = odeint(
                        lambda t, x: model(t, x),
                        input_tensor,
                        ts,
                        atol=1e-4,
                        rtol=1e-4,
                        method="euler",
                        condition=1,
                    )
                    # traj is [num_steps, 1, 2, 256, 256]
                    # We want the final step, channel 0 (predicted clean)
                    final_pred = traj[-1, 0, 0].cpu().numpy() # (256, 256)
                
                samples.append(final_pred)

            # Calculate MMSE (mean over samples)
            samples = np.stack(samples, axis=0) # (num_samples, 256, 256)
            mmse = np.mean(samples, axis=0)
            
            # Denormalize (Channel 0 for clean)
            prediction = denormalize(mmse, SUBSET_NAME, channel=0, path=None)
            
            # Convert to uint8
            prediction = np.clip(prediction, 0, 255).astype(np.uint8)
            
            # Save MMSE result
            out_img = Image.fromarray(prediction, mode='L')
            out_img.save(out / image_file)

if __name__ == "__main__":
    app()

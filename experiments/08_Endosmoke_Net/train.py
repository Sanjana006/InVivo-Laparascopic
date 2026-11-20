"""Train EndoSmoke-Net with paired conditional flow matching."""
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedLaparoscopyDataset
from models.endosmoke_net import EndoSmokeNet
from utils import seed_everything, pseudo_density_target, save_image


def run_epoch(model, loader, optimizer, device, scaler, train=True, density_weight=0.25):
    model.train(train)
    total, flow_total, den_total, count = 0.0, 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            smoky = batch["smoky"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            b = smoky.size(0)
            x0 = torch.randn_like(clean)
            t = torch.rand(b, device=device)
            t_img = t[:,None,None,None]
            x_t = (1-t_img)*x0 + t_img*clean
            target_velocity = clean - x0
            density_target = pseudo_density_target(smoky, clean)
            if train: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device.type=="cuda")):
                velocity, density = model(x_t, smoky, t)
                flow_loss = F.mse_loss(velocity, target_velocity)
                density_loss = F.smooth_l1_loss(density, density_target)
                loss = flow_loss + density_weight*density_loss
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer); scaler.update()
            total += loss.item()*b
            flow_total += flow_loss.item()*b
            den_total += density_loss.item()*b
            count += b
    return total/count, flow_total/count, den_total/count


def save_training_visuals(model, dataset, device, output_dir, epoch, num_samples=4, steps=10, solver="euler"):
    """Save a small fixed validation preview after each selected epoch."""
    if len(dataset) == 0 or num_samples <= 0:
        return
    model.eval()
    epoch_dir = Path(output_dir) / "training_visuals" / f"epoch_{epoch:03d}"
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            item = dataset[idx]
            smoky = item["smoky"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)
            prediction, density = model.sample(smoky, steps=steps, seed=epoch * 1000 + idx, solver=solver)
            pseudo = pseudo_density_target(smoky, clean)
            stem = Path(item["name"]).stem
            save_image(smoky[0], epoch_dir / "smoky" / f"{stem}_smoky.png")
            save_image(prediction[0], epoch_dir / "cleaned" / f"{stem}_cleaned.png")
            save_image(clean[0], epoch_dir / "ground_truth" / f"{stem}_clean.png")
            save_image(density[0].repeat(3, 1, 1) * 2 - 1,
                       epoch_dir / "density_predicted" / f"{stem}_density.png")
            save_image(pseudo[0].repeat(3, 1, 1) * 2 - 1,
                       epoch_dir / "density_pseudo_target" / f"{stem}_pseudo_density.png")
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/laparoscopy")
    p.add_argument("--output_dir", default="experiments/08_Endosmoke_Net/outputs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--density_weight", type=float, default=0.25)
    p.add_argument("--endovit_checkpoint", default=None)
    p.add_argument("--no_endovit", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--save_visuals_every", type=int, default=1,
                   help="Save validation image previews every N epochs; 0 disables it.")
    p.add_argument("--num_visual_samples", type=int, default=4,
                   help="Number of validation pairs to save per preview epoch.")
    p.add_argument("--visual_steps", type=int, default=10,
                   help="ODE sampling steps used for training previews.")
    p.add_argument("--visual_solver", choices=["euler", "rk4"], default="euler")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_dir)
    train_ds = PairedLaparoscopyDataset(root/"train_crop", args.image_size, augment=True)
    val_ds = PairedLaparoscopyDataset(root/"val_crop", args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type=="cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type=="cuda"))
    model = EndoSmokeNet(
        endovit_checkpoint=args.endovit_checkpoint, use_endovit=not args.no_endovit,
        freeze_endovit=True).to(device)
    trainable = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    start_epoch, best = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)+1
        best = ckpt.get("best_val_loss", best)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(start_epoch, args.epochs+1):
        tr = run_epoch(model, train_loader, optimizer, device, scaler, True, args.density_weight)
        va = run_epoch(model, val_loader, optimizer, device, scaler, False, args.density_weight)
        scheduler.step()
        row = {"epoch":epoch, "train_loss":tr[0], "train_flow":tr[1], "train_density":tr[2],
               "val_loss":va[0], "val_flow":va[1], "val_density":va[2],
               "lr":optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps(row))
        torch.save({"epoch":epoch, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                    "best_val_loss":best, "args":vars(args)}, out/"last_checkpoint.pth")
        if va[0] < best:
            best = va[0]
            torch.save({"epoch":epoch, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                        "best_val_loss":best, "args":vars(args)}, out/"best_model.pth")
        with (out/"history.json").open("w") as f: json.dump(history, f, indent=2)
        if args.save_visuals_every > 0 and epoch % args.save_visuals_every == 0:
            save_training_visuals(model, val_ds, device, out, epoch,
                                  num_samples=args.num_visual_samples,
                                  steps=args.visual_steps, solver=args.visual_solver)
    print(f"Training complete. Best validation loss: {best:.6f}")


if __name__ == "__main__": main()

#!/bin/bash
#SBATCH --job-name=EndoSmokeNet
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=experiments/08_Endosmoke_Net/logs/train_%j.out
#SBATCH --error=experiments/08_Endosmoke_Net/logs/train_%j.err

echo "Running on: $(hostname)"
date

module load anaconda3-2025-12-2
module load cuda-13.0

source experiments/08_Endosmoke_Net/.venv/bin/activate

nvidia-smi

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

python -u experiments/08_Endosmoke_Net/train.py \
    --data_root data/laparoscopy \
    --image_size 224 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4

echo "Training script finished."
date

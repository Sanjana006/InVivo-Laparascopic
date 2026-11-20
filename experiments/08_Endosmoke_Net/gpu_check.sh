#!/bin/bash
#SBATCH --job-name=endosmoke_gpu_check
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=gpu_check_%j.out
#SBATCH --error=gpu_check_%j.err

echo "Running on: $(hostname)"

module load anaconda3-2025-12-2
module load cuda-13.0

source .venv/bin/activate

echo "Checking NVIDIA GPU..."
nvidia-smi

echo "Checking PyTorch CUDA..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"

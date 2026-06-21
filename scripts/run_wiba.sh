#!/bin/bash
#SBATCH --job-name=wiba_pipeline
#SBATCH --account=PAS2119
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=/fs/scratch/PAS2882/integin1212/wiba/logs/wiba_%j.out
#SBATCH --error=/fs/scratch/PAS2882/integin1212/wiba/logs/wiba_%j.err

# Usage:
#   sbatch run_wiba.sh defense
#   sbatch run_wiba.sh econ
#   sbatch run_wiba.sh tech
#   sbatch run_wiba.sh all

DATASET=${1:-all}

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wiba

# Reduce GPU memory fragmentation (helps avoid OOM on smaller-VRAM GPUs like V100)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /fs/scratch/PAS2882/integin1212/wiba/logs

echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Dataset:   $DATASET"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"
echo "=============================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"

python /fs/scratch/PAS2882/integin1212/wiba/scripts/wiba_pipeline.py "$DATASET"

echo "=============================="
echo "End: $(date)"
echo "=============================="

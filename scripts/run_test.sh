#!/bin/bash
#SBATCH --job-name=wiba_test
#SBATCH --account=PAS2119
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/scratch/PAS2882/integin1212/wiba/logs/wiba_test_%j.out
#SBATCH --error=/fs/scratch/PAS2882/integin1212/wiba/logs/wiba_test_%j.err

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wiba

mkdir -p /fs/scratch/PAS2882/integin1212/wiba/logs

echo "=============================="
echo "TEST Job ID:  $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "Start:        $(date)"
echo "=============================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python /fs/scratch/PAS2882/integin1212/wiba/scripts/test_pipeline.py \
    --dataset all \
    --n 5

echo "=============================="
echo "End: $(date)"
echo "=============================="

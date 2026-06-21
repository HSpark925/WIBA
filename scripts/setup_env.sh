#!/bin/bash
# Creates the 'wiba' conda environment and installs required packages.
# Run ONCE from a login node:
#   bash /fs/scratch/PAS2882/integin1212/wiba/scripts/setup_env.sh

set -e

module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME="wiba"
ENV_DIR="/users/PAS2119/integin1212/miniconda3/envs/${ENV_NAME}"  # path only, not used directly

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Hugging Face stack..."
pip install transformers>=4.40.0 peft>=0.10.0 accelerate>=0.27.0 bitsandbytes>=0.43.0

echo "Installing data utilities..."
pip install pyreadr pandas numpy tqdm nltk sentencepiece

echo "Verifying key packages..."
python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import peft; print('peft', peft.__version__)"
python -c "import pyreadr; print('pyreadr ok')"
python -c "import bitsandbytes; print('bitsandbytes ok')"

echo ""
echo "Setup complete. Activate with:"
echo "  module load miniconda3/24.1.2-py310 && conda activate wiba"

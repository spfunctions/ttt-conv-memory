#!/bin/bash
# setup.sh — bare-metal Linux GPU host install (Ubuntu 22.04 + CUDA 12.x)
# For Modal-based runs, see modal_app.py instead.
set -euxo pipefail

# 1. System packages
apt-get update
apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    python3.11 python3.11-dev python3.11-venv python3-pip

# 2. Python venv pinned at 3.11
python3.11 -m venv /opt/ttt-venv
# shellcheck disable=SC1091
source /opt/ttt-venv/bin/activate
pip install --upgrade pip wheel setuptools

# 3. PyTorch first, with CUDA 12.8 wheel
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Flash-attention prebuilt wheel
# Adjust the wheel URL if the architecture differs. Linux x86_64 / cu12 / torch2.8 / cxx11abiTRUE / cp311.
pip install flash-attn==2.8.3 --no-build-isolation

# 5. Remaining deps
pip install -r requirements.txt

# 6. Clone In-Place TTT
mkdir -p /opt/repos
cd /opt/repos
if [ ! -d In-Place-TTT ]; then
    git clone https://github.com/ByteDance-Seed/In-Place-TTT.git
fi
cd In-Place-TTT
# Pin to known-good commit (will update once we verify)
git rev-parse HEAD

# 7. Smoke test
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B'); print('Qwen3 tokenizer OK')"

echo "Setup complete. Activate with: source /opt/ttt-venv/bin/activate"

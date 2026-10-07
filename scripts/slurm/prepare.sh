#!/usr/bin/env bash
# One-time environment setup: Python env with all dependencies, the paper's code,
# the training data, and the model weights in the Hugging Face cache.
# Needs internet. Run it on the login node, or submit scripts/slurm/prepare.sbatch
# (which also checks the result on a GPU node).
#
#   bash scripts/slurm/prepare.sh
#
# Settings (environment variables):
#   MODEL     model to download (default Qwen/Qwen3.5-4B)
#   VENV      where to create the environment (default ./.venv). torch + vLLM need ~10 GB,
#             so put it (and the repo) on project/scratch storage if home has a small quota.
#   HF_HOME   Hugging Face cache for the weights (~9 GB for the 4B model)
#   PY        Python to use (must be 3.10-3.13). If the system python3 is older, uv is
#             installed in ~/.local/bin and fetches Python 3.12 by itself.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm won't create the --output directory
# cluster defaults (scripts/slurm/site.env), if present
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
VENV=${VENV:-.venv}
# keep pip/uv caches next to the environment rather than in a small home directory
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$(dirname "$(realpath -m "$VENV")")/.cache/pip}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$(dirname "$(realpath -m "$VENV")")/.cache/uv}

py_ok() { "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; }

if [ ! -x "$VENV/bin/python" ]; then
  PY=${PY:-python3}
  if py_ok "$PY"; then
    echo "creating $VENV with $($PY --version)"
    "$PY" -m venv "$VENV"
  else
    echo "$PY is missing or not 3.10-3.13; using uv to get Python 3.12"
    command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12 --seed "$VENV"
  fi
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip
./setup.sh gpu

for t in lstar ttt; do
  [ -f "data/think_v1/$t/think/train.chat.jsonl" ] || \
    python scripts/generate_data.py --per-n 300 --teacher "$t" --scaffold think,thought --out "data/think_v1/$t"
done
STAGES=data NGPU=1 ./scripts/run_open_model.sh   # verifies the data against data/reference_stats/

echo "Hugging Face cache: ${HF_HOME:-~/.cache/huggingface}"
df -h "${HF_HOME:-$HOME}" 2>/dev/null | tail -1 || true
python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
# weights only (skip duplicate formats); resumes a partial download
print("model cached at", snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "merges.txt", "vocab.*"]))
EOF

python - <<'EOF'
import importlib
for m in ["torch", "transformers", "peft", "vllm", "fla"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m:12s} MISSING ({e})")
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print("transformers knows qwen3_5:", "qwen3_5" in CONFIG_MAPPING)
EOF
echo "prepared. Next: sbatch -p <partition> scripts/slurm/smoke.sbatch"

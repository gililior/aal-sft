#!/usr/bin/env bash
# One-time preparation, run on the LOGIN node (compute nodes often have no internet):
# creates .venv, installs everything, fetches the paper's code, generates the data,
# and downloads the model weights into the Hugging Face cache.
#
#   bash scripts/slurm/prepare.sh
#   MODEL=Qwen/Qwen3.5-9B bash scripts/slurm/prepare.sh
#
# If your cluster uses environment modules, load Python/CUDA first, e.g.
#   module load python/3.11 cuda/12.4
# and point HF_HOME at shared storage with enough space (the 4B model is ~9 GB):
#   export HF_HOME=/path/to/shared/hf_cache
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # Slurm won't create the --output directory

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
PY=${PY:-python3}

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
./setup.sh gpu

for t in lstar ttt; do
  [ -f "data/think_v1/$t/think/train.chat.jsonl" ] || \
    python scripts/generate_data.py --per-n 300 --teacher "$t" --scaffold think,thought --out "data/think_v1/$t"
done
STAGES=data ./scripts/run_open_model.sh   # verifies the data against data/reference_stats/

python - "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("model cached at", p)
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
echo "prepared. Next: sbatch [your -p/-A/--gres flags] scripts/slurm/smoke.sbatch"

#!/usr/bin/env bash
# Environment setup as a job (for clusters where heavy installs on the login node
# are discouraged), followed by a check on the GPU itself. Needs internet on the node.
#
#   mkdir -p logs && sbatch scripts/slurm/prepare.sbatch
#
#SBATCH --job-name=aal-prepare
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
# cluster defaults (scripts/slurm/site.env), if present
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi
if [ -n "${PRE_CMD:-}" ]; then eval "$PRE_CMD"; fi

bash scripts/slurm/prepare.sh

# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"
python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
try:
    import fla  # noqa: F401
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
    print("flash-linear-attention kernels importable")
except Exception as e:
    print("WARNING flash-linear-attention:", e)
EOF
echo "environment ready"

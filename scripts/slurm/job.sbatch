#!/usr/bin/env bash
# Generic GPU job: runs scripts/run_open_model.sh with whatever STAGES / TEACHERS /
# TAG / ... are exported into it. Submitted by smoke.sbatch's instructions and
# submit_all.sh; cluster-specific flags (-p, -A, --gres type, --qos) go on the
# sbatch command line, which overrides the defaults below.
#
#SBATCH --job-name=aal-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
# cluster defaults (scripts/slurm/site.env), if present
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi

# Optional cluster setup, e.g. PRE_CMD="module load cuda/12.4"
if [ -n "${PRE_CMD:-}" ]; then eval "$PRE_CMD"; fi
# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}       # weights were cached by prepare.sh
export TOKENIZERS_PARALLELISM=false
# a per-job port so several jobs can share a node
export PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}

echo "job ${SLURM_JOB_ID:-?} on $(hostname): STAGES='${STAGES:-}' TEACHERS='${TEACHERS:-}' TAG='${TAG:-}'"
nvidia-smi -L || true
./scripts/run_open_model.sh

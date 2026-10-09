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
# vLLM's FlashInfer top-k/top-p sampler JIT-compiles kernels and needs nvcc, which
# compute nodes may not have; vLLM's own sampler needs nothing extra.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
# If other kernels still need nvcc, use a CUDA toolkit from modules or a standard path.
if ! command -v nvcc > /dev/null 2>&1; then
  if type module > /dev/null 2>&1; then module load cuda > /dev/null 2>&1 || true; fi
fi
if ! command -v nvcc > /dev/null 2>&1; then
  for d in ${CUDA_HOME:-} /usr/local/cuda /usr/local/cuda-* /opt/cuda /usr/lib/cuda /usr/local/nvidia/cuda; do
    if [ -x "$d/bin/nvcc" ]; then export CUDA_HOME=$d PATH=$d/bin:$PATH; break; fi
  done
fi
echo "nvcc: $(command -v nvcc || echo 'not found (fine unless vLLM needs to compile kernels)')"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation
# a per-job port so several jobs can share a node
export PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}

echo "job ${SLURM_JOB_ID:-?} on $(hostname): STAGES='${STAGES:-}' TEACHERS='${TEACHERS:-}' TAG='${TAG:-}'"
echo "GPUs visible: ${CUDA_VISIBLE_DEVICES:-none}"
./scripts/run_open_model.sh

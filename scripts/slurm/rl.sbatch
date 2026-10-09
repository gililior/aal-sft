#!/usr/bin/env bash
# GRPO on one node: vLLM rollouts on the first GPU, LoRA updates on the rest.
# Resubmitting resumes from the last finished step; when all steps are done, the
# final adapter is evaluated on the paper's 160 instances (native per-turn format).
#
#   sbatch --export=ALL,NAME=rl_base scripts/slurm/rl.sbatch
#   sbatch --export=ALL,NAME=rl_sft,INIT_ADAPTER=runs/qwen3.5-4b-pt-ttt/final scripts/slurm/rl.sbatch
#
# Knobs (env): MODEL, TAG, NAME, INIT_ADAPTER, STEPS, DFAS_PER_STEP, GROUP_SIZE,
# TURNS_PER_EPISODE (0 = every turn), MAX_NEW_TOKENS, LR, CURRICULUM_START, RL_ARGS (extra flags).
#
#SBATCH --job-name=aal-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi
if [ -n "${PRE_CMD:-}" ]; then eval "$PRE_CMD"; fi
# shellcheck disable=SC1091
source "${VENV:-.venv}/bin/activate"

MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TAG=${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')-pt}
NAME=${NAME:?set NAME, e.g. rl_base or rl_sft}
OUT=runs/$TAG-$NAME
STEPS=${STEPS:-200}
PORT=${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [ "${#GPUS[@]}" -lt 2 ]; then echo "need at least 2 GPUs (1 rollout + 1 update)"; exit 1; fi
ROLL_GPU=${GPUS[0]}
TRAIN_GPUS=$(IFS=','; echo "${GPUS[*]:1}")
NTRAIN=$(( ${#GPUS[@]} - 1 ))
echo "job ${SLURM_JOB_ID:-?} on $(hostname): $OUT | rollout GPU $ROLL_GPU | update GPUs $TRAIN_GPUS"

mkdir -p logs "$OUT"
VLOG=logs/vllm_rl_${NAME}_${SLURM_JOB_ID:-local}.log
CUDA_VISIBLE_DEVICES=$ROLL_GPU vllm serve "$MODEL" --port "$PORT" --max-model-len 32768 \
    --language-model-only --enable-lora --max-lora-rank 64 --max-loras 1 \
    --gpu-memory-utilization 0.9 > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 240); do
  curl -sf "http://localhost:$PORT/v1/models" > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited; see $VLOG"; exit 1; }
  sleep 5
done

args=(scripts/train_rl.py --model "$MODEL" --out "$OUT" --vllm-url "http://localhost:$PORT" --steps "$STEPS")
[ -n "${INIT_ADAPTER:-}" ] && args+=(--init-adapter "$INIT_ADAPTER")
[ -n "${DFAS_PER_STEP:-}" ] && args+=(--dfas-per-step "$DFAS_PER_STEP")
[ -n "${GROUP_SIZE:-}" ] && args+=(--group-size "$GROUP_SIZE")
[ -n "${TURNS_PER_EPISODE:-}" ] && args+=(--turns-per-episode "$TURNS_PER_EPISODE")
[ -n "${MAX_NEW_TOKENS:-}" ] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")
[ -n "${LR:-}" ] && args+=(--lr "$LR")
[ -n "${CURRICULUM_START:-}" ] && args+=(--curriculum-start "$CURRICULUM_START")
# shellcheck disable=SC2206
[ -n "${RL_ARGS:-}" ] && args+=($RL_ARGS)

if [ ! -f "$OUT/final/adapter_model.safetensors" ]; then
  if [ "$NTRAIN" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --standalone --nproc_per_node "$NTRAIN" "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS python "${args[@]}"
  fi
fi

# training finished: evaluate the final adapter (paper's 160 instances, native format)
kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true
if [ -f "$OUT/final/adapter_model.safetensors" ] && [ "${SKIP_EVAL:-0}" != 1 ]; then
  CUDA_VISIBLE_DEVICES=$ROLL_GPU NGPU=1 TRAJ_MODE=per-turn TAG=$TAG TEACHERS=$NAME STAGES=eval \
    ./scripts/run_open_model.sh
fi

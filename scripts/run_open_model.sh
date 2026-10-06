#!/usr/bin/env bash
# End-to-end open-model pipeline on a GPU VM:
#   setup -> data -> eval untuned base -> LoRA-train on L* and on TTT -> eval each.
#
#   ./scripts/run_open_model.sh                 # everything
#   STAGES="train eval" ./scripts/run_open_model.sh
#   TEACHERS=ttt EPOCHS=2 ./scripts/run_open_model.sh
#
# Results: results/<model-tag>-{base,lstar,ttt}/summary.json ; adapters: runs/<model-tag>-<teacher>/final
# Rough cost (4B, one H100 80GB): ~80M training tokens per epoch per teacher, ~4-5 h/epoch.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-4B-Thinking-2507}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
TEACHERS=${TEACHERS:-"lstar ttt"}
STAGES=${STAGES:-"setup data base train eval"}
EPOCHS=${EPOCHS:-1}
MQ_TURNS=${MQ_TURNS:-8}          # MQ turns sampled per trajectory (all EQ turns kept)
LORA_RANK=${LORA_RANK:-32}
MAX_LEN=${MAX_LEN:-16384}
PORT=${PORT:-8000}
EVAL_WORKERS=${EVAL_WORKERS:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-40960}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
TAG=${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo -e "\n=== $(date '+%F %T') $* ===" ; }

if has setup; then
  log setup
  ./setup.sh gpu
fi

if has data; then
  for t in $TEACHERS; do
    if [ ! -f "data/think_v1/$t/think/train.chat.jsonl" ]; then
      log "generating $t data"
      python scripts/generate_data.py --per-n 300 --teacher "$t" --scaffold think,thought --out "data/think_v1/$t"
    fi
    # same counts as the reference build (data/reference_stats/<t>.json) or stop
    python - "$t" <<'EOF'
import json, sys
t = sys.argv[1]
got = json.load(open(f"data/think_v1/{t}/stats.json"))["scaffolds"]["think"]
ref = json.load(open(f"data/reference_stats/{t}.json"))["scaffolds"]["think"]
keys = ["examples", "val_examples", "assistant_turns"]
diff = {k: (got[k], ref[k]) for k in keys if got[k] != ref[k]}
if diff:
    sys.exit(f"{t}: generated data differs from reference build: {diff}")
print(f"{t}: data matches reference build ({got['examples']} trajectories, {got['assistant_turns']} turns)")
EOF
  done
fi

VLLM_PID=""
stop_vllm() { if [ -n "$VLLM_PID" ]; then kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""; fi; }
trap stop_vllm EXIT

serve() {  # serve [adapter_dir]
  local extra=()
  if [ -n "${1:-}" ]; then
    extra=(--enable-lora --max-lora-rank "$LORA_RANK" --lora-modules "aal=$1")
  fi
  mkdir -p logs
  vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAX_MODEL_LEN" \
      --reasoning-parser "$REASONING_PARSER" --tensor-parallel-size "${NGPU:-1}" \
      "${extra[@]}" > "logs/vllm_$(date +%s).log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then return 0; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM exited; see logs/"; exit 1; fi
    sleep 5
  done
  echo "vLLM did not come up in 15 min"; exit 1
}

evaluate() {  # evaluate <served-model-name> <out>
  python scripts/evaluate.py --backend openai --model "$1" --base-url "http://localhost:$PORT/v1" \
      --scaffold think --workers "$EVAL_WORKERS" --out "$2"
}

if has base; then
  log "eval untuned $MODEL"
  serve
  evaluate "$MODEL" "results/$TAG-base"
  stop_vllm
fi

for t in $TEACHERS; do
  if has train; then
    log "train on $t"
    args=(scripts/train_hf_lora.py --model "$MODEL" --data "data/think_v1/$t/think" --out "runs/$TAG-$t"
          --epochs "$EPOCHS" --mq-turns-per-traj "$MQ_TURNS" --rank "$LORA_RANK" --alpha $((2 * LORA_RANK))
          --max-len "$MAX_LEN")
    if [ "${NGPU:-1}" -gt 1 ]; then
      torchrun --nproc_per_node "$NGPU" "${args[@]}"
    else
      python "${args[@]}"
    fi
  fi
  if has eval; then
    log "eval $t-tuned"
    serve "runs/$TAG-$t/final"
    evaluate aal "results/$TAG-$t"
    stop_vllm
  fi
done

log done
python scripts/compare_results.py results/$TAG-* || true

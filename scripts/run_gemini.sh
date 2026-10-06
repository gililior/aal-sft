#!/usr/bin/env bash
# Gemini pipeline (visible <THOUGHT> reasoning), run from any machine with gcloud:
#   data -> two Vertex tuning jobs (L*, TTT) in parallel -> eval base + both tuned models.
#
#   gcloud auth login && gcloud auth application-default login
#   PROJECT=my-proj BUCKET=my-bucket ./scripts/run_gemini.sh
#
# Knobs: BASE_MODEL (gemini-3.5-flash), LOCATION (us-central1), EPOCHS (3), TEACHERS, STAGES, EVAL_WORKERS.
# Tuning and inference are billed to PROJECT.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PROJECT:?set PROJECT}"
: "${BUCKET:?set BUCKET (created in LOCATION if missing)}"
BASE_MODEL=${BASE_MODEL:-gemini-3.5-flash}
LOCATION=${LOCATION:-us-central1}
EPOCHS=${EPOCHS:-3}
TEACHERS=${TEACHERS:-"lstar ttt"}
STAGES=${STAGES:-"setup data base tune eval"}
EVAL_WORKERS=${EVAL_WORKERS:-8}
TAG=${TAG:-$BASE_MODEL}

has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo -e "\n=== $(date '+%F %T') $* ===" ; }

if has setup; then
  log setup
  ./setup.sh
  gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" > /dev/null 2>&1 \
    || gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$LOCATION"
fi

if has data; then
  for t in $TEACHERS; do
    [ -f "data/think_v1/$t/thought/train.gemini.jsonl" ] || \
      python scripts/generate_data.py --per-n 300 --teacher "$t" --scaffold think,thought --out "data/think_v1/$t"
  done
fi

evaluate() {  # evaluate <model-or-endpoint> <out>
  python scripts/evaluate.py --backend vertex --model "$1" --project "$PROJECT" --location "$LOCATION" \
      --scaffold thought --thinking-level MINIMAL --workers "$EVAL_WORKERS" --out "$2"
}

if has base; then
  log "eval untuned $BASE_MODEL (thought scaffold, thinking MINIMAL)"
  evaluate "$BASE_MODEL" "results/$TAG-base-thought" &
fi

if has tune; then
  mkdir -p logs
  pids=()
  for t in $TEACHERS; do
    log "tuning on $t"
    python scripts/train_gemini_vertex.py --project "$PROJECT" --location "$LOCATION" --bucket "$BUCKET" \
        --data "data/think_v1/$t/thought" --base-model "$BASE_MODEL" --epochs "$EPOCHS" \
        --name "aal-$t-thought" > "logs/tune_$t.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
fi

if has eval; then
  for t in $TEACHERS; do
    f="data/think_v1/$t/thought/aal-$t-thought.tuned.json"
    ep=$(python -c "import json;print(json.load(open('$f'))['endpoint'])")
    log "eval $t-tuned ($ep)"
    evaluate "$ep" "results/$TAG-$t-thought"
  done
fi
wait
python scripts/compare_results.py results/$TAG-* results/replay_* || true

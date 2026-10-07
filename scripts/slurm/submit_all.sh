#!/usr/bin/env bash
# Submit the full experiment as three independent GPU jobs that run in parallel:
#   base eval  |  train+eval on L*  |  train+eval on TTT
# When all three finish:  python scripts/compare_results.py results/<tag>-* results/replay_*
#
#   bash scripts/slurm/submit_all.sh                  # add SBATCH_ARGS="-p ... -A ..." if needed
#
# GPUs: TRAIN_GRES (default 4x L40S per training job, trained data-parallel with
# torchrun; eval then serves on one of them) and EVAL_GRES for the base eval.
# Time limits: TRAIN_TIME / EVAL_TIME. Training checkpoints every ~5% of an epoch,
# so if a job hits its time limit, just run this script again: finished parts are
# skipped and training resumes from the latest checkpoint.
#
# Other knobs are passed through to run_open_model.sh: MODEL, EPOCHS, MQ_TURNS,
# LORA_RANK, PRE_CMD, ...
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
# cluster defaults (scripts/slurm/site.env), if present
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi

SBATCH_ARGS=${SBATCH_ARGS:-}
TRAIN_GRES=${TRAIN_GRES:-gpu:l40s:4}
EVAL_GRES=${EVAL_GRES:-gpu:l40s:1}
TRAIN_TIME=${TRAIN_TIME:-12:00:00}
EVAL_TIME=${EVAL_TIME:-6:00:00}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TAG=${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
COMMON="ALL,MODEL=$MODEL,TAG=$TAG"

# shellcheck disable=SC2086
if [ -f "results/$TAG-base/summary.json" ]; then
  base="(done)"
else
  base=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-base" --gres="$EVAL_GRES" --time="$EVAL_TIME" \
         --export="$COMMON,STAGES=base" scripts/slurm/job.sbatch)
fi
ids=("$base")
for t in lstar ttt; do
  # shellcheck disable=SC2086
  if [ -f "results/$TAG-$t/summary.json" ]; then ids+=("(done)"); continue; fi
  id=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-$t" --gres="$TRAIN_GRES" --time="$TRAIN_TIME" \
       --cpus-per-task=32 --mem=160G \
       --export="$COMMON,STAGES=train+eval,TEACHERS=$t" scripts/slurm/job.sbatch)
  ids+=("$id")
done
echo "submitted: base=$base lstar=${ids[1]} ttt=${ids[2]}"
echo "watch:  squeue -u \$USER ;  tail -f logs/slurm-aal-$TAG-*.out"
echo "after:  source ${VENV:-.venv}/bin/activate && python scripts/compare_results.py results/$TAG-* results/replay_*"

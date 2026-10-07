#!/usr/bin/env bash
# Submit the full experiment as three independent GPU jobs that run in parallel:
#   base eval  |  train+eval on L*  |  train+eval on TTT
# When all three finish:  python scripts/compare_results.py results/<tag>-* results/replay_*
#
#   SBATCH_ARGS="-p gpu -A my_account --gres=gpu:a100:1" bash scripts/slurm/submit_all.sh
#
# Other knobs are passed through to run_open_model.sh: MODEL, EPOCHS, MQ_TURNS,
# LORA_RANK, PRE_CMD, ...
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

SBATCH_ARGS=${SBATCH_ARGS:-}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TAG=${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')}
COMMON="ALL,MODEL=$MODEL,TAG=$TAG"

# shellcheck disable=SC2086
base=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-base" --time=6:00:00 \
       --export="$COMMON,STAGES=base" scripts/slurm/job.sbatch)
ids=("$base")
for t in lstar ttt; do
  # shellcheck disable=SC2086
  id=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-$t" --time=16:00:00 \
       --export="$COMMON,STAGES=train+eval,TEACHERS=$t" scripts/slurm/job.sbatch)
  ids+=("$id")
done
echo "submitted: base=$base lstar=${ids[1]} ttt=${ids[2]}"
echo "watch:  squeue -u \$USER ;  tail -f logs/slurm-aal-$TAG-*.out"
echo "after:  source .venv/bin/activate && python scripts/compare_results.py results/$TAG-* results/replay_*"

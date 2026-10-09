#!/usr/bin/env bash
# The RL experiment (all in Qwen's native per-turn format, tag <model>-pt):
#   1. base-model eval                      (1 GPU)   results/<tag>-base
#   2. per-turn SFT on TTT, then its eval   (4 GPUs)  runs/<tag>-ttt/final, results/<tag>-ttt
#   3. RL from the base model               (4 GPUs)  runs/<tag>-rl_base, results/<tag>-rl_base
#   4. RL from the SFT adapter, after 2     (4 GPUs)  runs/<tag>-rl_sft,  results/<tag>-rl_sft
# Rerun this script to resubmit whatever hasn't finished (jobs resume where they stopped).
#
#   bash scripts/slurm/submit_rl.sh
#   SFT_TEACHER=lstar STEPS=300 bash scripts/slurm/submit_rl.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
if [ -f scripts/slurm/site.env ]; then source scripts/slurm/site.env; fi

SBATCH_ARGS=${SBATCH_ARGS:-}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
TAG=${TAG:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')-pt}
SFT_TEACHER=${SFT_TEACHER:-ttt}
STEPS=${STEPS:-200}
COMMON="ALL,MODEL=$MODEL,TAG=$TAG,TRAJ_MODE=per-turn"
done_() { [ -f "$1" ]; }

# shellcheck disable=SC2086
if done_ "results/$TAG-base/summary.json"; then base="(done)"; else
  base=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-base" --gres=gpu:l40s:1 --time=12:00:00 \
         --export="$COMMON,STAGES=base" scripts/slurm/job.sbatch); fi

sft_dep=""
# shellcheck disable=SC2086
if done_ "runs/$TAG-$SFT_TEACHER/final/adapter_model.safetensors"; then sft="(done)"; else
  sft=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-$SFT_TEACHER" --gres=gpu:l40s:4 --time=24:00:00 \
        --cpus-per-task=32 --mem=160G \
        --export="$COMMON,STAGES=train+eval,TEACHERS=$SFT_TEACHER,EPOCHS=1" scripts/slurm/job.sbatch)
  sft_dep="--dependency=afterok:$sft"; fi

# shellcheck disable=SC2086
if done_ "results/$TAG-rl_base/summary.json"; then rlb="(done)"; else
  rlb=$(sbatch --parsable $SBATCH_ARGS --job-name="aal-$TAG-rl_base" \
        --export="ALL,MODEL=$MODEL,TAG=$TAG,NAME=rl_base,STEPS=$STEPS" scripts/slurm/rl.sbatch); fi

# shellcheck disable=SC2086
if done_ "results/$TAG-rl_sft/summary.json"; then rls="(done)"; else
  rls=$(sbatch --parsable $SBATCH_ARGS $sft_dep --job-name="aal-$TAG-rl_sft" \
        --export="ALL,MODEL=$MODEL,TAG=$TAG,NAME=rl_sft,STEPS=$STEPS,INIT_ADAPTER=runs/$TAG-$SFT_TEACHER/final" \
        scripts/slurm/rl.sbatch); fi

echo "submitted: base-eval=$base sft=$sft rl_base=$rlb rl_sft=$rls"
echo "progress: tail -f logs/slurm-aal-rl-*.out   (one line per RL step; full stats in runs/$TAG-rl_*/rl_log.jsonl)"
echo "table:    source ${VENV:-.venv}/bin/activate && python scripts/compare_results.py results/$TAG-*"

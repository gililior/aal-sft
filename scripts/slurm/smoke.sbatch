#!/usr/bin/env bash
# ~30 min end-to-end check of every stage on a tiny subset, before the long runs:
# eval the base model on 4 instances, LoRA-train on 20 TTT trajectories, serve the
# adapter and eval it on the same 4 instances. Results land in results/smoke-*.
#
#   sbatch -p <partition> -A <account> scripts/slurm/smoke.sbatch
#
#SBATCH --job-name=aal-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
rm -rf runs/smoke-ttt results/smoke-base results/smoke-ttt

export TAG=smoke STAGES="base train eval" TEACHERS=ttt
export TRAIN_LIMIT=20 EPOCHS=1 EVAL_NSTATES=2-3 EVAL_SEEDS=1-2 EVAL_WORKERS=4
exec bash scripts/slurm/job.sbatch

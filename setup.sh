#!/usr/bin/env bash
# Fetch the paper's code at the commit this pipeline was verified against and
# install the core dependencies (data generation + evaluation harness).
#   ./setup.sh          core only
#   ./setup.sh gpu      + vLLM / transformers / peft for open-model training
set -euo pipefail
cd "$(dirname "$0")"

UPSTREAM_COMMIT=b8b93e215cb2d6fdb6ef921759610baa348cc520
if [ ! -d upstream_repo ]; then
  git clone https://github.com/reefmenaged/Agentic_Automata_Learning upstream_repo
fi
git -C upstream_repo fetch --quiet origin "$UPSTREAM_COMMIT" || true
git -C upstream_repo checkout --quiet "$UPSTREAM_COMMIT"
ln -sfn upstream_repo/app upstream_app

# AALpy and automata-lib pinned to the exact commits the datasets were built with
# (L* query sequences depend on AALpy's defaults). pyvis is not needed: it is stubbed.
pip install -r requirements.txt
if [ "${1:-}" = "gpu" ]; then
  pip install -r requirements-gpu.txt
fi
echo "setup done"

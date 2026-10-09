# SFT on classic automata-learning trajectories

Fine-tune an LLM to imitate L* / TTT on the *agentic automata learning* task
(Menaged et al., arXiv:2606.16576), then evaluate it with the paper's own harness.

```
setup.sh                       clone the paper's code (pinned commit) + deps
aal_sft/oracle_session.py      oracle that emits exactly the harness's prompts / TOOL_RESULTs
aal_sft/teacher.py             L*/TTT run -> verified chat trajectory (stateless or stateful)
aal_sft/models.py              Vertex Gemini / OpenAI-compatible / teacher-replay adapters
scripts/generate_data.py       build train/val sets (Vertex + chat JSONL)
scripts/train_gemini_vertex.py launch a Vertex supervised tuning job
scripts/train_hf_lora.py       LoRA SFT for an open model (assistant-only loss)
scripts/evaluate.py            paper benchmark (160 instances) via the upstream runtime
tests/test_fidelity_vs_upstream_runtime.py
```

## On a Slurm cluster

Cluster defaults live in `scripts/slurm/site.env` (environment location, GPU types,
time limits); anything set in your shell overrides them.

```bash
cd /cs/snapless/gabis/gililior && git clone https://github.com/gililior/aal-sft.git && cd aal-sft
mkdir -p logs
sbatch scripts/slurm/prepare.sbatch     # env in virtual_envs/aal_sft, data, weights, GPU check
sbatch scripts/slurm/smoke.sbatch       # ~30 min, every stage on a tiny subset (results/smoke-*)
bash scripts/slurm/submit_all.sh        # base eval (1x L40S) + L* and TTT train+eval (4x L40S each)
```

`prepare.sh` uses the system `python3` if it is 3.10–3.13; otherwise it installs
`uv` and fetches Python 3.12. If compute nodes have no internet, run
`bash scripts/slurm/prepare.sh` on the login node instead of `prepare.sbatch`.
Jobs run with `HF_HUB_OFFLINE=1` from the cache filled by `prepare.sh`. Training
checkpoints about 20 times per epoch: if a job is killed, run `submit_all.sh`
again. Finished parts are skipped and training resumes from the latest
checkpoint. Logs are in `logs/slurm-*.out`.

## RL (GRPO with the oracle as reward)

```bash
# smoke test: 2 RL steps on small DFAs, 2 GPUs (1 rollout + 1 update), no eval
sbatch --gres=gpu:l40s:2 --time=2:00:00 \
  --export=ALL,NAME=rl_smoke,STEPS=2,DFAS_PER_STEP=2,GROUP_SIZE=4,SKIP_EVAL=1 scripts/slurm/rl.sbatch

# the experiment: base eval, per-turn SFT on TTT, RL from base, RL from SFT
bash scripts/slurm/submit_rl.sh
```

- **Format:** Qwen's native chat format: each turn sees every earlier query and
  oracle answer, but not earlier reasoning. Same harness messages as the
  benchmark. Code: `aal_sft/rl_env.py`, `scripts/train_rl.py`.
- **Reward:** 1 if the final equivalence query is correct, else 0.
- **Credit:** advantage = reward minus the group mean over 8 episodes of the
  same DFA (Dr. GRPO), applied to every turn (`--turns-per-episode` subsamples).
  Loss normalized by a constant, so long and short turns aren't reweighted.
- **Steps:** 8 DFAs × 8 episodes, one on-policy update. Curriculum starts at ≤4
  states and adds a size when the largest one passes 50% twice in a row.
  Training DFAs never equal an eval DFA.
- **Hardware:** vLLM on GPU 0 with the adapter hot-loaded each step
  (`/v1/load_lora_adapter`); updates on the remaining GPUs with DDP.
- **Resuming and output:** the state is saved every step, so a resubmit
  resumes. Per-step stats go to `runs/<tag>-<name>/rl_log.jsonl`. When training
  finishes, the job evaluates `final/` on the 160 instances.
- **Tests:** `tests/test_rl_env.py` checks the episode loop against the real
  oracle with a scripted policy.

Decisions and next steps are tracked in `docs/decision_log.md`.

## Open thinking model on a GPU VM (one command)

```bash
git clone <this repo> && cd <repo>
./scripts/run_open_model.sh        # setup, data, base eval, train+eval on L* and on TTT
```

Stages and knobs are environment variables (`STAGES`, `TEACHERS`, `MODEL`,
`EPOCHS`, `MQ_TURNS`, `NGPU`, ...; see the script header). Defaults:

* **Model:** Qwen/Qwen3.5-4B, which thinks by default (`MODEL=Qwen/Qwen3.5-9B`
  also works). Training uses its text-only class; vLLM serves it with
  `--language-model-only` and the `qwen3` reasoning parser.
* **LoRA:** rank 32, 1 epoch. Targets cover the attention, MLP *and* Gated
  DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `out_proj`). Only 1 in 4 of
  Qwen3.5's token-mixing layers is standard attention, so the usual
  q/k/v/o-only recipe would leave most of the model untouched. After training,
  adapter weights are renamed to the vision-language checkpoint's
  `model.language_model.*` naming, which is what vLLM expects.
* **Kernels:** `flash-linear-attention` provides fast Gated DeltaNet kernels;
  without it transformers falls back to a slow pure-PyTorch path (the training
  script warns).
* **Eval sampling:** temperature 0.6, top_p 0.95, top_k 20 (Qwen's thinking-mode
  setting for precise tasks; greedy decoding can loop in thinking models).

Expect about 80M training tokens per epoch per teacher, roughly 4–5 h/epoch on
one H100. The data is regenerated on the VM (about 2 min) and checked against
`data/reference_stats/`. At the end, `scripts/compare_results.py` prints the
base / L* / TTT table.

`tests/test_masking_qwen3_template.py` checks the loss masks against the real
Qwen3 and Qwen3.5 chat templates (template files from llama.cpp `models/templates/`).

## Gemini with visible reasoning (one command)

```bash
gcloud auth login && gcloud auth application-default login
PROJECT=my-proj BUCKET=my-bucket ./scripts/run_gemini.sh
```

This tunes gemini-3.5-flash on the L* and TTT `thought` datasets in parallel,
evaluates the untuned base model and both tuned endpoints (thought scaffold,
thinking MINIMAL), and prints the comparison table. Tuning and inference are
billed to PROJECT.

## Quick start

```bash
./setup.sh
python scripts/generate_data.py --per-n 300 --scaffold both --out data/v1      # ~1 min on 2 cores

# Gemini on Vertex
python scripts/train_gemini_vertex.py --project P --bucket B --data data/v1/stateless \
    --base-model gemini-3.5-flash --epochs 3
python scripts/evaluate.py --backend vertex --model <endpoint printed above> --project P \
    --out results/gemini35-sft-stateless

# Baselines to compare against (same harness)
python scripts/evaluate.py --backend vertex --model gemini-3.5-flash --project P \
    --thinking-level MINIMAL --out results/gemini35-base-minimal
python scripts/evaluate.py --backend teacher --out results/teacher   # must be 100%, Δ=0
```

## How the data is made

For each sampled target DFA (the paper's Boltzmann sampler, binary alphabet) the
upstream L* and TTT implementations are run against the paper's oracle
(deterministic short counterexamples). The teacher is whichever used fewer
queries on that DFA (`--teacher lstar|ttt` forces one). Its logged queries are
replayed through the same tool classes the harness uses, and every oracle answer
is asserted to equal what the algorithm saw; the trajectory must end in a
correct EQ. The budget shown in the prompt is the paper's: 2× the better of L*/TTT.

**Fidelity.** `tests/test_fidelity_vs_upstream_runtime.py` drives the real
`LLM_interactive_game` with a scripted model emitting our assistant turns: every
message it sends is byte-identical to ours, and it reports the DFA solved.

**Held-out set.** Training seeds start at 100000. Any DFA equivalent to one of
the paper's 160 eval targets (seeds 1–20 × n=2..9) is dropped, as are duplicates.
Only 10 non-eval 2-state DFAs exist, so n=2 is small by necessity.
Train/val split is by DFA.

**v1 dataset** (`--per-n 300`): 2,110 trajectories per scaffold, 79,804 assistant
turns, 0 replay failures. TTT is the teacher for 79% (rising from 59% at n=3 to
91% at n=9; L* wins at n=2). Stateless examples average ~1.8k tokens at n=3 and
~5.7k at n=9; stateful ~18.7k at n=9, max ~34k (Vertex limit: 131k).

## Reasoning data: one run on L*, one on TTT

`aal_sft/reasoning.py` runs AALpy 2.0.0 L* (Rivest–Schapire counterexample
processing) and the paper's TTT with tracing hooks. Whenever a query actually
reaches the oracle, the call stack tells which algorithm step issued it, and
that step's state becomes the rationale. For L* that's the observation table
with the cell being filled, why a row was moved to S, or the RS binary-search
split. For TTT it's the access words, the discrimination tree, the sift path,
the RS split, or the discriminator search. The traced runs reproduce the
reference query sequences exactly (asserted).

Note: the paper's "TTT" is a discrimination-tree learner (Kearns–Vazirani style
with RS counterexample analysis and a discriminator search), not full TTT with
discriminator finalization. The TTT student imitates that implementation.

```bash
for t in lstar ttt; do
  python scripts/generate_data.py --per-n 300 --teacher $t --scaffold think,thought --out data/think_v1/$t
done
# open thinking model (reasoning in the native thinking channel)
python scripts/train_hf_lora.py --model Qwen/Qwen3.5-4B \
    --data data/think_v1/ttt/think --out runs/qwen3.5-4b-ttt            # and .../lstar/think
vllm serve Qwen/Qwen3.5-4B --language-model-only --enable-lora --max-lora-rank 32 \
    --reasoning-parser qwen3 --lora-modules aal=runs/qwen3.5-4b-ttt/final --max-model-len 40960
python scripts/evaluate.py --backend openai --model aal --scaffold think --out results/qwen-think-ttt
# Gemini with visible reasoning (<THOUGHT> block, built-in thinking MINIMAL)
python scripts/train_gemini_vertex.py --project P --bucket B --data data/think_v1/ttt/thought
python scripts/evaluate.py --backend vertex --model <endpoint> --project P --scaffold thought \
    --out results/gemini-thought-ttt
```

* **think**: the prompt is the paper's stateless prompt, and reasoning goes in
  `reasoning_content`. Default training (`--trajectory-mode full`): each
  trajectory is one example with loss on every model turn (reasoning and query),
  and earlier turns' reasoning stays in context. Qwen's own template drops
  reasoning after each user message, and the oracle's answers arrive as user
  messages, so training and serving both use `templates/qwen_keep_reasoning.jinja`
  (vLLM `--chat-template`), and the eval client puts each turn's reasoning back
  into the history (`--keep-reasoning`). The vocabulary projection is applied in
  2,048-row chunks with recomputation, so 40k-token trajectories fit on a 48 GB
  GPU. `--trajectory-mode per-turn` is the alternative: one example per selected
  turn, with earlier reasoning dropped as in Qwen's own format.
* **thought**: the prompt asks for `<THOUGHT>` then `<TOOL_ACTION>`, and earlier
  thoughts stay in context, so whole trajectories are trained. Examples are at
  most ~30k tokens.

Both teachers use the same 2,110 training DFAs (L*: 97k assistant turns; TTT: 81k).

**Ceilings under the paper's harness** (teacher replayed on all 160 instances):

| bucket | L* success | L* Δ calls | TTT success | TTT Δ calls |
|---|---|---|---|---|
| 2–3 | 100 | +0.3 | 100 | +1.9 |
| 4–5 | 100 | +4.6 | 100 | +0.7 |
| 6–7 | 100 | +7.8 | 100 | +1.0 |
| 8–9 | 97.5 | +16.8 | 100 | +0.8 |

Δ is relative to the better of L*/TTT per instance. One instance (n=8, seed 1:
L* 96 calls, budget 82) is unsolvable for a perfect L* imitator.

## Scaffolds

* **stateless**: assistant turns are a bare `<TOOL_ACTION>` block, exactly the
  released harness's format.
* **stateful**: the paper's GOAL / WORKING_MEMORY / THOUGHT / TOOL_ACTION format.
  This prompt is not in the public repo; it is reconstructed from Appendix A.1.2
  (same rules as the released prompt, structured output policy swapped in). The
  WORKING_MEMORY and THOUGHT text is *synthesized*: accepted/rejected words, last
  hypothesis and counterexample, plus a templated rationale. It does not expose
  the algorithm's internal state (observation table / discrimination tree).

## Things to keep in mind when reading results

* **Gemini 3.8 Flash may not be tunable.** Google's tuning docs list Gemini 3.5
  Flash and 3.1 Flash-Lite; 3.8 Flash isn't listed. The launcher defaults to
  `gemini-3.5-flash`; pass `--base-model gemini-3.8-flash` to try.
* **Thinking.** Gemini SFT trains without thinking traces, and tuned models
  should run at `thinking_level=MINIMAL`. The paper's baselines ran at HIGH. To
  isolate the effect of SFT, compare against the *same base model at MINIMAL*
  in addition to the paper's numbers.
* **Action-only imitation.** In the stateless format the model must reconstruct
  the algorithm's internal state from the transcript at every step. That is
  exactly the failure mode the paper reports, so it is the interesting test, but
  a richer stateful target (serialized observation table / discrimination tree)
  is a natural ablation.
* **Exposure bias.** Teacher trajectories never contain mistakes, so the model
  never sees recovery. If off-trajectory drift dominates failures, consider
  DAgger-style data (restart L*/TTT seeded with the model's observations) or the
  SDK's reinforcement-tuning option with success/efficiency reward.

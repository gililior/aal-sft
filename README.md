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

## Open thinking model on a GPU VM (one command)

```bash
git clone <this repo> && cd <repo>
./scripts/run_open_model.sh        # setup, data, base eval, train+eval on L* and on TTT
```

Stages and knobs are environment variables (`STAGES`, `TEACHERS`, `MODEL`,
`EPOCHS`, `MQ_TURNS`, `NGPU`, ...; see the script header). Defaults are
Qwen/Qwen3-4B-Thinking-2507, 1 epoch, rank-32 LoRA, served with vLLM's `qwen3`
reasoning parser and evaluated with the `think` scaffold. Expect about 80M
training tokens per epoch per teacher, roughly 4–5 h/epoch on one H100. The data
is regenerated on the VM (about 2 min) and checked against
`data/reference_stats/`. At the end, `scripts/compare_results.py` prints the
base / L* / TTT table.

`tests/test_masking_qwen3_template.py` checks the loss masks against the real
Qwen3 chat template (template file from llama.cpp `models/templates/`).

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
python scripts/train_hf_lora.py --model Qwen/Qwen3-4B-Thinking-2507 \
    --data data/think_v1/ttt/think --out runs/qwen3-4b-think-ttt        # and .../lstar/think
vllm serve Qwen/Qwen3-4B-Thinking-2507 --enable-lora --reasoning-parser qwen3 \
    --lora-modules aal=runs/qwen3-4b-think-ttt/final --max-model-len 32768
python scripts/evaluate.py --backend openai --model aal --scaffold think --out results/qwen-think-ttt
# Gemini with visible reasoning (<THOUGHT> block, built-in thinking MINIMAL)
python scripts/train_gemini_vertex.py --project P --bucket B --data data/think_v1/ttt/thought
python scripts/evaluate.py --backend vertex --model <endpoint> --project P --scaffold thought \
    --out results/gemini-thought-ttt
```

* **think**: the prompt is the paper's stateless prompt, and reasoning goes in
  `reasoning_content`. Thinking models drop earlier turns' reasoning at
  inference, so training is per turn: visible history plus that turn's reasoning
  and action, with loss on that turn only. All EQ turns are kept, plus
  `--mq-turns-per-traj` (default 8) sampled MQ turns per trajectory.
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

# multiagent-se

`multiagent-se` is an experimental Python framework for comparing four LLM
workflow configurations on software development tasks while holding the task
inputs and model settings constant:

1. `baseline`: a single-call system reference,
2. `naive`: a free-form planner-to-coder handoff,
3. `structured_no_validation`: a structured but unvalidated handoff,
4. `contract`: schema validation with a bounded planner retry.

`structured_no_validation` and `contract` use the same planner and coder path.
The validator node and bounded retry are the only interventions that differ
between them; `tests/test_arm_equivalence.py` enforces this invariant.

The code covers the three studies reported in the accompanying paper:
Study 1A (EvalPlus, Gemini 3.5 Flash Lite and DeepSeek V4 Flash), Study 1B
(the same EvalPlus tasks, Gemini and GPT-5.6 Luna), and Study 2
(BigCodeBench-Hard, Gemini and Luna). See
[Follow-up studies](#follow-up-studies-study-1b-and-study-2).

This public distribution includes the source code, frozen task definitions,
deterministic tests, and the supplementary material of the accompanying paper
(see [Supplementary material](#supplementary-material)). It does not
distribute raw model outputs, manuscript drafts, or internal working notes.

## Setup

Requirements: Python 3.12+, Git, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/lsaliherenl/multiagent-se-public.git
cd multiagent-se-public
uv sync
```

For live model calls, copy the example environment file and populate only your
local `.env` file:

```bash
cp .env.example .env
```

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Git ignores `.env`, logs, key and certificate files, and publication
workspaces. Never paste an API key into a command line, commit message, or
error output.

## Quick verification

The test suite requires neither network access nor a real API key:

```bash
uv run pytest -q
```

To regenerate both task sets from their sources:

```bash
uv run python scripts/fetch_tasks.py
uv run python scripts/fetch_evalplus.py
```

The second command downloads version-pinned EvalPlus sources, verifies their
SHA-256 checksums, applies the eligibility filters, and deterministically
rebuilds `tasks_heldout/`.

## Running an experiment

Validate the workflow on the small pilot set first:

```bash
uv run python -m eval.runner \
  --name local_smoke \
  --model dev \
  --task-set pilot \
  --tasks 2 \
  --repeats 1
```

A held-out run intentionally requires explicit model, task-set, and repetition
arguments:

```bash
uv run python -m eval.runner \
  --name heldout_run \
  --model main \
  --task-set heldout \
  --repeats 3
```

Outputs are written under `logs/exp_<name>/` and are excluded from version
control. The runner records task hashes, the prompt contract, the environment,
and the Git commit in its manifest. Start a full experimental run only from a
clean, committed working tree.

To analyse one run (one model on one task set):

```bash
uv run python -m analysis.analyze --exp logs/exp_heldout_run
```

## Follow-up studies (Study 1B and Study 2)

Follow-up runs use the same runner with an explicit `--study` identity. The
second producer is selected by its role name, `followup_secondary`
(GPT-5.6 Luna):

```bash
uv run python -m eval.runner --name study1b_gemini --study study1b \
  --model main --task-set heldout --repeats 3
uv run python -m eval.runner --name study1b_luna --study study1b \
  --model followup_secondary --task-set heldout --repeats 3
```

Study 2 scores code inside the official BigCodeBench evaluation image, so it
needs Docker. Build the pinned image and the read-only NLTK resource volume
once (the build downloads several gigabytes):

```bash
uv run python scripts/bigcodebench_runtime.py build
uv run python scripts/bigcodebench_runtime.py hydrate
uv run python scripts/bigcodebench_runtime.py verify
```

Then run the two Study 2 cells:

```bash
uv run python -m eval.runner --name study2_gemini --study study2 \
  --model main --task-set study2_complex --repeats 3
uv run python -m eval.runner --name study2_luna --study study2 \
  --model followup_secondary --task-set study2_complex --repeats 3
```

The runner checks the image and the resource volume before any model call,
and records the image ID in the manifest. A fresh build can have a different
image ID from the paper's run (apt packages are not pinned); the manifest
states whether it matches.

Each cell is analysed separately; the cross-cell estimands (moderation
between regimes and the Study 1A to 1B bridge) come from
`analysis/followup.py`:

```bash
uv run python -m analysis.followup \
  --study1a-gemini logs/exp_heldout_run \
  --study1b-gemini logs/exp_study1b_gemini --study1b-luna logs/exp_study1b_luna \
  --study2-gemini logs/exp_study2_gemini --study2-luna logs/exp_study2_luna
```

`reproduction/paper_run_identity.json` records the identity of the paper's
held-out runs (models, request policy, prompt-contract hash, task-file
hashes, evaluator image). `tests/test_followup.py` checks that this code
reproduces every field the code determines. The spend ledger, health gates,
and authorization records that governed the original runs are not part of
this distribution; they did not affect request content
(`EXPERIMENT_PROTOCOL.md` §13).

## Repository structure

| Path | Contents |
| --- | --- |
| `agents/` | Planner, coder, tester, and validator roles |
| `pipeline/` | LangGraph workflows for the four experimental arms |
| `eval/` | Sandbox, harness, runner, result contract, and MAST tooling |
| `analysis/` | Task-level analysis and exploratory uncertainty tooling |
| `uncertainty/` | Self-consistency measurement |
| `tasks/` | 20-task development and pilot set |
| `tasks_heldout/` | 50-task EvalPlus-derived held-out set (Studies 1A and 1B) |
| `tasks_followup_dev/` | 16-task BigCodeBench-Hard development set (Study 2) |
| `tasks_study2_complex/` | 50-task BigCodeBench-Hard held-out set (Study 2) |
| `docker/bigcodebench/` | Pinned BigCodeBench evaluation image recipe |
| `reproduction/` | Identity of the paper's held-out runs (no results) |
| `supplement/` | Supplementary material of the paper |
| `scripts/` | Task generation, evaluator runtime, health-check, and compliance utilities |
| `tests/` | Offline, deterministic regression tests |

See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md) for the experimental
design and interpretation boundaries, and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for dataset licensing.

## Security warning

This project executes model-generated Python code in a subprocess. Its timeout,
output limit, and environment-variable allowlist do not constitute a strong
security sandbox; in particular, they do not provide memory or CPU isolation
on Windows. Do not run untrusted code on a machine that contains sensitive data
or network credentials. Use a disposable container or a separate virtual
machine when stronger isolation is required.

## Supplementary material

[`supplement/Supplementary_Material.pdf`](supplement/Supplementary_Material.pdf)
is the supplementary material of the paper "Structured Inter-Agent Handoffs
and Validation: A Four-Arm Mechanism Study in Multi-Agent Code Generation". It
contains Tables S1 to S13 and Figures S1 to S6, which the paper cites. The
tables report results from the paper's frozen analysis; they were not
regenerated from this repository. Some tables and figures print the code arm
identifiers listed at the top of this README; the paper uses reader labels
(`baseline` = Single-call, `naive` = NL handoff, `structured_no_validation` =
JSON handoff, `contract` = Contract).

## Scope of results

The source code itself makes no performance or effect claims. Results
obtained with the code should be analyzed separately for each model and each
study; repetitions must not be treated as independent samples. Findings
generalize only to the EvalPlus-derived, equality-compatible subset (Studies
1A and 1B) and to the selected BigCodeBench-Hard subset (Study 2).

## License

The project's original source code is released under the
[Apache License 2.0](LICENSE). Third-party benchmark content remains subject to
the separate terms documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

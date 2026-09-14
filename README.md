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

This public distribution includes the source code, frozen task definitions,
and deterministic tests. It does not distribute raw model outputs,
experimental results, manuscript drafts, publication figures, or internal
working notes.

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

## Repository structure

| Path | Contents |
| --- | --- |
| `agents/` | Planner, coder, tester, and validator roles |
| `pipeline/` | LangGraph workflows for the four experimental arms |
| `eval/` | Sandbox, harness, runner, result contract, and MAST tooling |
| `analysis/` | Task-level analysis and exploratory uncertainty tooling |
| `uncertainty/` | Self-consistency measurement |
| `tasks/` | 20-task development and pilot set |
| `tasks_heldout/` | 50-task EvalPlus-derived held-out set |
| `scripts/` | Task generation, health-check, and compliance utilities |
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

## Scope of results

This public source distribution makes no performance or effect claims. Results
obtained with the code should be analyzed separately for the two provider
models; repetitions must not be treated as independent samples, and findings
should be generalized only to the EvalPlus-derived, equality-compatible
subset.

## License

The project's original source code is released under the
[Apache License 2.0](LICENSE). Third-party benchmark content remains subject to
the separate terms documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

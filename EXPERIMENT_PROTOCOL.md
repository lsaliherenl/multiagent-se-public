# Public experiment protocol

This document records the design constraints required to run or extend the
public codebase. It intentionally contains no observed results, manuscript
text, claim ledger, or publication schedule.

## 1. Objective

Measure how communication structure and contract enforcement affect an
LLM-based software-engineering workflow while keeping tasks, producer models,
prompts, decoding parameters, and evaluation machinery fixed across arms.

## 3. Arms

| Arm | Planner-to-coder handoff | Validator | Planner retry |
| --- | --- | --- | --- |
| `baseline` | No multi-agent handoff | No | No |
| `naive` | Free-form text | No | No |
| `structured_no_validation` | Canonical structured envelope | No | No |
| `contract` | The same canonical structured envelope | Yes | Bounded |

The `contract` versus `structured_no_validation` comparison isolates the
validator plus bounded retry. Their planner and coder implementations must be
identical. The `structured_no_validation` versus `naive` comparison isolates
the structured representation without enforcement. The `contract` versus
`naive` comparison is a total-package contrast and must not be described as a
validator-only effect.

## 5. Task sets

- `tasks/`: 20 development tasks (12 HumanEval and 8 sanitized MBPP). These
  may be used for debugging and calibration only.
- `tasks_heldout/`: 50 version-pinned EvalPlus-derived tasks (30 HumanEval+
  and 20 MBPP+) selected without model calls.

Development and held-out tasks are stored separately and must never be pooled.
The held-out set uses an equality-compatible oracle; therefore results do not
represent the complete HumanEval+ or MBPP+ benchmarks.

## 4. Models and execution design

- Two producer models run the complete held-out design separately.
- Each producer uses the same 50 tasks, four arms, and three repeats.
- Arm order rotates deterministically within task/repeat blocks.
- Model, task hashes, prompt-contract hash, environment, dependency lock,
  decoding parameters, provider routing, and Git commit are recorded in the
  run manifest.
- A run may resume only when the stored manifest remains compatible. Previous
  `run_error` rows remain auditable but do not count as completed runs.

The full held-out design is 50 tasks x 4 arms x 3 repeats = 600 arm-runs per
producer model. Producer models must not be pooled in analysis.

## 8. Outcomes and inference

- Primary task outcome: Plus-test pass.
- Secondary task outcome: base-test pass.
- Repeats are repeated measurements of the same task, not independent samples.
- For each model and arm, first aggregate repeats within task.
- Contrasts use paired task-level differences.
- Uncertainty is estimated with a task-cluster bootstrap using 10,000
  iterations and a fixed seed.

Intervals that include zero are not evidence of equivalence or no effect.
Exploratory self-consistency and failure-label analyses must be reported as
exploratory and kept separate from the prespecified arm contrasts.

## 9. Failure labeling

Failed runs may be categorized with the MAST-compatible schema in `eval/`.
Model identity is withheld from labeling prompts where practical, producer
self-labels do not determine the final label, and any human-reviewed subset
must be described as a selected subset rather than a population-wide estimate.

## 7. Result records

Every completed arm-run is validated before append. Identity fields include
the producer model, arm, task, repeat, experiment, and run ID. Duplicate,
unexpected, malformed, and missing records are surfaced by the shared result
schema instead of being silently dropped.

## 12. Reproducibility and safety gates

1. Run `uv run pytest -q` without API access.
2. Use a clean, committed Git tree for any formal run.
3. Reproduce task files with the pinned download scripts and verify hashes.
4. Keep API keys only in `.env`; candidate code receives a restricted
   environment variable allowlist.
5. Treat the subprocess harness as a correctness boundary, not a security
   sandbox. Use a disposable container or VM for untrusted code.
6. Store generated runs outside version control under `logs/`.

## Public/private boundary

The public repository contains implementation, tests, task definitions, and
this protocol. It excludes raw model outputs, result datasets, manuscript
sources, generated paper assets, internal notes, unpublished follow-up study
code, and claim-specific analysis modules.

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

## 13. Follow-up studies (Study 1B and Study 2)

The original design above is Study 1A. Two follow-up studies reuse the same
four arms, prompts, planner, coder, validator, and result contract; they change
only the producer roster and, for Study 2, the task regime. Every follow-up run
is started with an explicit `--study` identity; without it the runner treats a
run as Study 1A and refuses the Study 2 task sets.

| Study | Task set | Regime | Producers |
| --- | --- | --- | --- |
| 1B | `tasks_heldout/` (the same 50 tasks as Study 1A) | EvalPlus | Gemini 3.5 Flash Lite, GPT-5.6 Luna |
| 2 | `tasks_study2_complex/` (50 tasks) | BigCodeBench-Hard | Gemini 3.5 Flash Lite, GPT-5.6 Luna |

Development sets (`tasks/` for Study 1B, `tasks_followup_dev/` with 16 tasks
for Study 2) are for technical calibration only and never enter the primary
statistics. DeepSeek (the Study 1A replication model) does not produce
follow-up data, and judge or adjudicator models produce no data in any study.
Study 3 was deferred and has no profile in this distribution.

**Study 2 task set.** The tasks come from the `v0.1.4` split of
BigCodeBench-Hard at a fixed dataset revision. Of 148 tasks, 6 were removed by
a rule fixed before task content was inspected, and a structural-complexity
filter admitted 66 of the remaining 142. The prespecified pool gate required
80 eligible tasks and therefore failed; two protocol amendments followed,
both before any Study 2 result was seen: the 66 eligible tasks were split with
a frozen seed into 16 development and 50 held-out tasks, and a human-annotated
edge-case criterion was turned from an eligibility gate into a descriptive
attribute. The selection pipeline itself is not distributed; the frozen task
files and their selection manifests are, and the runner records their hashes.

**Request policy.** The generation policy is the same as Study 1A
(temperature 0.2, 8,192 output tokens, reasoning enabled for every role and
arm), with one route-level exception: the GPT-5.6 Luna route does not accept
`temperature`, so the field is omitted from Luna requests (never sent as
`null`) and the manifest records `temperature_policy`. Luna is pinned to the
OpenAI Standard endpoint with fallbacks disabled; Gemini keeps the default
routing policy. In follow-up runs the LiteLLM transport retry is disabled and
the same number of attempts is made in a visible loop in `agents/llm.py`; the
request body is unchanged.

**Study 2 evaluation.** Candidate code is scored by the official BigCodeBench
`untrusted_check` inside a pinned evaluation image with `--network none`
(`docker/bigcodebench/`, built by `scripts/bigcodebench_runtime.py`). A
read-only NLTK stopwords volume is verified by manifest hash before every
evaluation. BigCodeBench has a single hidden test suite, so `plus_pass` is a
compatibility alias of the same result, there is no Base-to-Plus attrition
measurement, and hidden tests never enter result records. A fresh image build
can have a different image ID from the paper's run; the ID is recorded in
every Study 2 manifest and must not change during a run.

**Estimands.** Each model-study cell is analysed separately with the Study 1A
procedure (§8). The primary confirmatory estimand is `contract - naive` on
Plus pass in the Study 2 Gemini cell; the Study 2 Luna cell is its
replication. Secondary, descriptive estimands are the mechanism contrasts in
all follow-up cells, the moderation difference `Delta[Study 2] - Delta[Study
1B]` per model (the two regimes' tasks are resampled independently), and the
Study 1A to Study 1B Gemini bridge on the same 50 tasks (tasks are resampled
jointly). A complexity-slope estimand was gated on task overlap and was not
run. Regimes and models are never pooled, unresolved `run_error` rows block
the analysis, and no p-value is produced (`analysis/followup.py`).
Error-class shares are not compared across regimes, because the two
evaluators use different failure taxonomies.

**Not distributed.** The original runs were additionally governed by a spend
ledger, provider-health and price-drift gates, and per-phase authorization
records. These controlled cost and operations, not request content, and are
not part of this distribution. `reproduction/paper_run_identity.json` records
the identity of the paper's held-out runs, and
`tests/test_followup.py` checks that this code reproduces every field it
determines.

## Public/private boundary

The public repository contains implementation, tests, task definitions, the
paper's supplementary material, and this protocol. It excludes raw model
outputs, result datasets, manuscript sources, internal notes, and the
operational gating code of the original runs.

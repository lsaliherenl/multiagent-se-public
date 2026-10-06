# multiagent-se — contributor guidance

## Authoritative public sources

1. `EXPERIMENT_PROTOCOL.md` defines the four-arm design and analysis limits.
2. `README.md` defines setup, execution, and repository scope.
3. `config.py` is the single source for executable constants.

## Invariants

- The arms are `baseline`, `naive`, `structured_no_validation`, and
  `contract`.
- `structured_no_validation` and `contract` differ only by the validator node
  and bounded retry. Do not create separate planner/coder implementations.
- `tasks/` is development-only. `tasks_heldout/` is the 50-task evaluation
  set; never pool the two.
- Study 1B and Study 2 runs require `--study`. `tasks_followup_dev/` is
  development-only; `tasks_study2_complex/` is the Study 2 evaluation set.
  Task files are frozen: their hashes are checked against
  `reproduction/paper_run_identity.json`.
- Route-level request exceptions (for example, no `temperature` for Luna) live
  only in `config.effective_request_policy`; agent code never compares slugs.
- Study 2 evaluation runs only inside the pinned BigCodeBench image with
  `--network none`; hidden tests never enter result records.
- Analyze producer models separately and use task-level inference.
- Generated code must run in `eval/sandbox.py`, never in the main process.
- Keep constants in `config.py`; agent and pipeline modules must not duplicate
  them.
- Tests must remain deterministic and API-free: `uv run pytest -q`.
- Never commit `.env`, API keys, run logs, model outputs, manuscripts, or
  internal research notes. The only distributed result material is the
  paper's supplementary PDF under `supplement/`.

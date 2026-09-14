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
- Analyze producer models separately and use task-level inference.
- Generated code must run in `eval/sandbox.py`, never in the main process.
- Keep constants in `config.py`; agent and pipeline modules must not duplicate
  them.
- Tests must remain deterministic and API-free: `uv run pytest -q`.
- Never commit `.env`, API keys, run logs, model outputs, manuscripts, paper
  figures, result tables, or internal research notes.

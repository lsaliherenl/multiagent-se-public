# Third-party notices

This repository's original source code is licensed under the Apache License
2.0. The project license does not replace the licenses that apply to the
redistributed programming-task material derived from the projects below.

## HumanEval

- Source: <https://github.com/openai/human-eval>
- Pinned source commit used by `scripts/fetch_tasks.py`:
  `463c980b59e818ace59f6f9803cd92c749ceae61`
- License: MIT License
- Copyright: OpenAI
- Included license copy: `LICENSES/HumanEval-MIT.txt`

The HumanEval-derived task prompts, tests, and reference solutions in `tasks/`
remain subject to the upstream MIT terms.

## MBPP

- Source: <https://github.com/google-research/google-research/tree/master/mbpp>
- Pinned source commit used by `scripts/fetch_tasks.py`:
  `f82046ba5aabbbb427dbfd38a254d26bff08b533`
- Dataset license: [Creative Commons Attribution 4.0 International
  (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)

The sanitized MBPP-derived material in `tasks/` remains subject to CC BY 4.0.
Attribution should identify Google Research and the MBPP dataset.

## EvalPlus, HumanEval+, and MBPP+

- Project: <https://github.com/evalplus/evalplus>
- HumanEval+ release: <https://github.com/evalplus/humanevalplus_release>
- MBPP+ release: <https://github.com/evalplus/mbppplus_release>
- License: Apache License 2.0
- Included license copy: `LICENSES/EvalPlus-Apache-2.0.txt`

The `tasks_heldout/` subset is generated from version-pinned HumanEval+ v0.1.10
and MBPP+ v0.2.0 release artifacts. It also contains material derived from the
underlying HumanEval and MBPP datasets, so the corresponding upstream terms
above continue to apply.

## Scope

See `tasks/README.md`, `tasks_heldout/README.md`, and
`tasks_heldout/_selection_manifest.json` for transformation, selection,
version, checksum, and source-URL details. This notice is an attribution and
scope record, not legal advice.

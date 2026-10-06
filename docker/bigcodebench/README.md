# BigCodeBench evaluation runtime (Study 2)

Study 2 scores candidate code with the official BigCodeBench evaluator,
`bigcodebench.eval.untrusted_check`, inside this image. Build and check it
with:

```bash
uv run python scripts/bigcodebench_runtime.py build
uv run python scripts/bigcodebench_runtime.py hydrate
uv run python scripts/bigcodebench_runtime.py verify
```

`build` clones BigCodeBench at the pinned commit into `.context/` (ignored by
Git) and builds the image as `multiagent-se/bigcodebench-official:g4v1`.
`hydrate` creates the read-only NLTK stopwords volume; it is the only step
that uses the network inside the container. Evaluation itself always runs
with `--network none` as the unprivileged `bigcodebenchuser` (UID 1000).

## Deviations from the official recipe

The recipe starts from the official `Docker/Evaluate.Dockerfile` of
BigCodeBench. No dependency content is changed; every deviation only removes a
mutable input or supports the evaluation channel.

| ID | Official | Here | Why |
| --- | --- | --- | --- |
| R1 | `FROM python:3.10-slim` | Base image pinned by digest | A tag is mutable |
| R2 | `pip install --upgrade pip` | `pip==24.0` | "Latest pip" is mutable; pip is a build tool, not an evaluator dependency |
| R3 | GitHub API cache-buster `ADD` | Removed | The source is already pinned |
| R4 | `git clone` of the default branch | `COPY` of a checkout at commit `09dd993f46c3fbf3a799465bb96d524edcb0b199`, verified during the build | Exact commit instead of a moving branch |
| R5 | `pip install -r` from the `main` branch URL | The same file's bytes at the pinned commit (`requirements-eval.canonical.txt`, 1,242 bytes), verified by SHA-256 and size | The branch URL is mutable; no line is added, removed, or changed |
| R6 | Dataset preload | Omitted | The task files are shipped in this repository |
| R7 | — | `/g6out` created and owned by `bigcodebenchuser` | Inputs and outputs move through a named volume with `docker cp`, so the evaluator can run unprivileged on Windows hosts as well |
| R8 | — | Fail-closed checks of the commit, the requirements hash and size, and the final user | A wrong input stops the build instead of producing a different image |

## Image identity

The base image is pinned, but `apt-get` packages are not, so a fresh build
can have a different image ID from the one used in the paper
(`sha256:3d80a30b7f8d2c511032afec5305bfa05f4e6e96db98459efc187814dd11c920`).
The runner records the image ID of every Study 2 run in its manifest, flags
whether it matches the paper's ID, and refuses to resume a run with a
different image. The image is also inspected before and after every
evaluation; a change stops the run.

## NLTK resource volume

Some tasks call `nltk.download('stopwords')`, which cannot succeed without a
network. The corpus is therefore downloaded once into the volume
`multiagent-se-nltk-stopwords-v1`, mounted read-only during evaluation, and
verified against a pinned manifest (35 files, 127,179 bytes, manifest SHA-256
`c06ae5113a8f80094b9cc93c7b23207a611c974e76cff7afc99c6934dcb0f871`) before
every evaluation. If the upstream corpus has changed, `hydrate` reports the
mismatch and Study 2 runs do not start.

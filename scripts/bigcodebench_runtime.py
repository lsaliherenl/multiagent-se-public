"""BigCodeBench evaluation runtime for Study 2: build, hydrate, verify.

Study 2 scores candidate code inside the official BigCodeBench evaluation image
(EXPERIMENT_PROTOCOL.md §13). This script prepares that image and the read-only
NLTK resource volume on a local Docker daemon. It makes no model call.

    uv run python scripts/bigcodebench_runtime.py build
    uv run python scripts/bigcodebench_runtime.py hydrate
    uv run python scripts/bigcodebench_runtime.py verify

build    Clones BigCodeBench at the pinned commit into a local build context,
         checks the pinned requirements bytes, and builds
         `docker/bigcodebench/Dockerfile` as `BIGCODEBENCH_IMAGE_TAG`.
         The base image is pinned by digest; apt layers are not, so the
         resulting image ID can differ from the one used in the paper. The ID
         is printed and recorded in every Study 2 run manifest.
hydrate  Creates the NLTK stopwords volume. This is the ONLY step that uses the
         network inside the container; evaluation itself runs with
         `--network none`. The volume is then verified against the pinned
         manifest hash; if upstream NLTK data has changed, verification fails
         and Study 2 runs refuse to start.
verify   Checks that the image exists and the resource volume matches.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    BIGCODEBENCH_FROZEN_COMMIT,
    BIGCODEBENCH_IMAGE_TAG,
    BIGCODEBENCH_OUTPUT_DIR,
    BIGCODEBENCH_PAPER_IMAGE_ID,
    BIGCODEBENCH_REQUIREMENTS_SHA256,
    BIGCODEBENCH_RESOURCE_VOLUME,
    BIGCODEBENCH_RUN_USER,
    BIGCODEBENCH_SOURCE_REPO,
    ROOT,
)
from eval import bigcodebench_backend as backend  # noqa: E402

RECIPE_DIR = ROOT / "docker" / "bigcodebench"
CONTEXT_DIR = RECIPE_DIR / ".context"
REQUIREMENTS = RECIPE_DIR / "requirements-eval.canonical.txt"


class RuntimeSetupError(RuntimeError):
    pass


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(argv, cwd=cwd, text=True, encoding="utf-8",
                          errors="replace", capture_output=True)
    if proc.returncode:
        raise RuntimeSetupError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr[-2000:]}")
    return proc


def check_requirements_bytes(path: Path = REQUIREMENTS) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BIGCODEBENCH_REQUIREMENTS_SHA256:
        raise RuntimeSetupError(
            f"{path.name} SHA-256 {digest} != pinned {BIGCODEBENCH_REQUIREMENTS_SHA256}")


def prepare_context(context: Path = CONTEXT_DIR) -> Path:
    """Fresh build context: Dockerfile, requirements, source at the pinned commit."""
    check_requirements_bytes()
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)
    shutil.copy2(RECIPE_DIR / "Dockerfile", context / "Dockerfile")
    shutil.copy2(REQUIREMENTS, context / REQUIREMENTS.name)
    src = context / "bigcodebench-src"
    _run(["git", "clone", "--quiet", BIGCODEBENCH_SOURCE_REPO, str(src)])
    _run(["git", "checkout", "--quiet", BIGCODEBENCH_FROZEN_COMMIT], cwd=src)
    head = _run(["git", "rev-parse", "HEAD"], cwd=src).stdout.strip()
    if head != BIGCODEBENCH_FROZEN_COMMIT:
        raise RuntimeSetupError(f"HEAD {head} != pinned {BIGCODEBENCH_FROZEN_COMMIT}")
    return context


def build() -> str:
    context = prepare_context()
    print(f"building {BIGCODEBENCH_IMAGE_TAG} (this takes a while) ...")
    subprocess.run(["docker", "build", "--platform", "linux/amd64",
                    "-t", BIGCODEBENCH_IMAGE_TAG, str(context)], check=True)
    image_id = backend._image_identity(backend.DockerChannel())
    print(f"image {BIGCODEBENCH_IMAGE_TAG} -> {image_id}")
    if image_id == BIGCODEBENCH_PAPER_IMAGE_ID:
        print("matches the image ID used in the paper.")
    else:
        print("differs from the paper's image ID "
              f"({BIGCODEBENCH_PAPER_IMAGE_ID}); this is expected for a fresh "
              "build and is recorded in every Study 2 manifest.")
    return image_id


def hydrate() -> dict:
    present = subprocess.run(["docker", "volume", "inspect", BIGCODEBENCH_RESOURCE_VOLUME],
                             capture_output=True)
    if present.returncode == 0:
        raise RuntimeSetupError(
            f"volume {BIGCODEBENCH_RESOURCE_VOLUME} already exists; run `verify`, or "
            f"remove it (docker volume rm {BIGCODEBENCH_RESOURCE_VOLUME}) to rebuild it")
    _run(["docker", "volume", "create", BIGCODEBENCH_RESOURCE_VOLUME])
    # Network is used here only, to fetch the corpus into the volume.
    _run(["docker", "run", "--rm", "--user", BIGCODEBENCH_RUN_USER,
          "-v", f"{BIGCODEBENCH_RESOURCE_VOLUME}:{BIGCODEBENCH_OUTPUT_DIR}",
          "--entrypoint", "python", BIGCODEBENCH_IMAGE_TAG,
          "-m", "nltk.downloader", "-d", BIGCODEBENCH_OUTPUT_DIR, "stopwords"])
    return verify()


def verify() -> dict:
    docker = backend.DockerChannel()
    image_id = backend._image_identity(docker)
    resource = backend.verify_resource_volume(docker)
    print(f"image ok: {image_id}")
    print(f"resource volume ok: {resource['manifest_sha256']}")
    return {"image_id": image_id, **resource}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "hydrate", "verify"))
    args = parser.parse_args(argv)
    try:
        {"build": build, "hydrate": hydrate, "verify": verify}[args.command]()
    except (RuntimeSetupError, backend.BigCodeBenchBackendError,
            subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

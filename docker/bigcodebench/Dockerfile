# BigCodeBench official evaluation runtime, pinned (EXPERIMENT_PROTOCOL.md §13).
# Based on the official Docker/Evaluate.Dockerfile (blob
# 8b2cdcd42fe14ab0f45dd18114f9ccf6f30bbbdb, sha256
# a0d56d681c9eeaf0c6d780309aab31b6df422bad21ad1a3de2c3da0598e05b12).
# Dependency CONTENT is unchanged; every deviation (R1-R8) is listed in
# docker/bigcodebench/README.md. Build with scripts/bigcodebench_runtime.py.
FROM python:3.10-slim@sha256:855690f49a018755f1f69d689a3ce5fd42eadd255754299eb440b5a1b647fc1f

# install git, g++ and python3-tk  [OFFICIAL, VERBATIM]
RUN apt-get update && apt-get install -y \
    git \
    g++ \
    python3-tk \
    zip \
    unzip \
    procps \
    r-base \
    libgdal-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# [R2] exact build-tool version instead of 'latest pip'
RUN python -m pip install pip==24.0

# [OFFICIAL, VERBATIM]
RUN adduser --disabled-password --gecos "" bigcodebenchuser
RUN rm -rf /bigcodebench

# [R3] cache-buster removed  [R4] exact commit instead of a mutable clone
COPY bigcodebench-src/ /bigcodebench/
COPY requirements-eval.canonical.txt /canonical/requirements-eval.txt

# [R8] fail-closed identity checks
RUN test -d /bigcodebench/.git \
    || (echo 'RECIPE_DEFECT: .git YOK' >&2; exit 90)
RUN git config --global --add safe.directory /bigcodebench \
    && HEAD_SHA="$(git -C /bigcodebench rev-parse HEAD)" \
    && [ "$HEAD_SHA" = "09dd993f46c3fbf3a799465bb96d524edcb0b199" ] \
    || (echo 'RECIPE_DEFECT: HEAD beklenen commit degil' >&2; exit 91)
RUN echo "a4d01fb12cbce5223b51f982265cb7975bea770b758cd85cc91b803d3293e39f  /canonical/requirements-eval.txt" | sha256sum -c - \
    || (echo 'RECIPE_DEFECT: requirements hash uyusmuyor' >&2; exit 92)
RUN [ "$(wc -c < /canonical/requirements-eval.txt)" = "1242" ] \
    || (echo 'RECIPE_DEFECT: requirements boyutu yanlis' >&2; exit 93)

# [OFFICIAL, VERBATIM]
RUN python -m pip install numpy==1.24.3 pyarrow==14.0.1

RUN cd /bigcodebench && \
    python -m pip install . --no-deps

RUN python -m pip install \
    appdirs \
    fire \
    multipledispatch \
    pqdm \
    tempdir \
    termcolor \
    tqdm \
    tree_sitter \
    tree-sitter-python \
    wget \
    transformers \
    datasets \
    gradio-client \
    numpy \
    rich \
    accelerate \
    anthropic \
    google-genai \
    mistralai \
    openai \
    e2b

# [R5] canonical blob bytes instead of the mutable main-branch URL
RUN python -m pip install -I --timeout 2000 -r /canonical/requirements-eval.txt

# Ensure the numpy version is compatible with the datasets version
RUN python -m pip install datasets==2.17.0

# [R6] dataset preload omitted (task files are shipped in this repository)

# Proof that installation completed (the evaluator is NOT run).
RUN python -c "import sys; v=sys.version_info[:2]; assert v==(3, 10), v; print('PY_OK', sys.version)"
RUN python -m pip show bigcodebench && echo 'EVALUATOR_INSTALLED_OK'

WORKDIR /app
RUN mkdir -p /app /g6out
RUN chown -R bigcodebenchuser:bigcodebenchuser /app /g6out
RUN chmod -R 777 /app

USER bigcodebenchuser
RUN [ "$(id -un)" = "bigcodebenchuser" ] \
    && test -w /g6out \
    && echo 'FINAL_USER_OK' \
    || (echo 'RECIPE_DEFECT: final kullanici/cikti dizini' >&2; exit 94)

ENTRYPOINT ["python3", "-m", "bigcodebench.evaluate"]
CMD ["sh", "-c", "pids=$(ps -u $(id -u) -o pid,comm | grep 'bigcodebench' | awk '{print $1}'); if [ -n \"$pids\" ]; then echo $pids | xargs -r kill; fi; rm -rf /tmp/*"]

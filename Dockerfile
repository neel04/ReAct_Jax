FROM python:3.11
ENV jax_threefry_partitionable=1

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      net-tools iproute2 procps ethtool \
      wget git gcc \
      neovim tmux \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN pip3 install --no-cache-dir uv && \
    uv venv "$VIRTUAL_ENV" --python 3.11
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install -U --no-cache-dir "jax[cuda12]"

RUN uv pip install -q --no-cache-dir \
    transformers datasets scalax tokenizers icecream wandb einops torch tqdm jaxtyping optuna equinox rich

RUN uv pip install -q --no-cache-dir -U optuna-integration plotly pdbpp

RUN uv pip install --no-cache-dir \
    git+https://github.com/neel04/lm-evaluation-harness.git@debug/mp

RUN uv pip install --no-cache-dir \
    git+https://github.com/google-deepmind/optax.git \
    git+https://github.com/deepmind/jmp \
    git+https://github.com/Findus23/jax-array-info.git

RUN uv pip install -q --no-cache-dir \
    tensorflow tensorboard-plugin-profile etils importlib_resources "cloud-tpu-profiler>=2.3.0"

WORKDIR /ReAct_Jax

CMD ["bash", "-lc", "tail -f /dev/null"]

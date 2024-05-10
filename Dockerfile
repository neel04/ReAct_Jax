FROM python:3.11

# Set environment variables
ENV WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f
ENV HF_TOKEN=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
ENV jax_threefry_partitionable=1

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y net-tools iproute2 procps ethtool && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install Ipython matplotlib
RUN pip3 install numpy pandas scipy

RUN pip3 install -U -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip3 install -q transformers datasets scalax tokenizers icecream wandb einops torch tqdm jaxtyping optax optuna equinox rich
RUN pip3 install tensorboard-plugin-profile comet-ml optuna-integration plotly
RUN pip3 install git+https://github.com/deepmind/jmp

WORKDIR /ReAct_Jax

# Set the entry point to bash 
ENTRYPOINT ["/bin/bash"]
#FROM ubuntu:latest
FROM python:3.11

ENV WANDB_API_KEY=78c7285b02548bf0c06dca38776c08bb6018593f

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install Ipython matplotlib
RUN pip3 install numpy pandas scipy

RUN pip3 install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip3 install -q datasets icecream tokenizers wandb einops torch tqdm jaxtyping optax equinox jaxlib
RUN pip3 uninstall tensorflow -y
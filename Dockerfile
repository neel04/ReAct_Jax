# Colab image
FROM europe-docker.pkg.dev/colab-images/public/runtime

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get install -y gcc && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER root
RUN pip3 install Ipython matplotlib wandb jax equinox jaxtyping optax
RUN pip3 install einops tqdm jupyterlab numpy pandas scipy
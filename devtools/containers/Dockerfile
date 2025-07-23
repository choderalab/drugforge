# Use the conda-forge base image with Python
FROM mambaorg/micromamba:jammy

LABEL org.opencontainers.image.source=https://github.com/choderalab/drugforge
LABEL org.opencontainers.image.description="drugforge toolkit"
LABEL org.opencontainers.image.licenses=MIT

# set environment variables
ENV PYTHONUNBUFFERED 1

RUN micromamba config append channels conda-forge
RUN micromamba config append channels openeye

COPY --chown=$MAMBA_USER:$MAMBA_USER  devtools/conda-envs/drugforge-ubuntu-latest.yml /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER  .  /home/mambauser/drugforge

RUN micromamba install -y -n base git -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1


WORKDIR /home/mambauser/drugforge

RUN python -m pip install --no-deps -e drugforge-alchemy
RUN python -m pip install --no-deps -e drugforge-cli
RUN python -m pip install --no-deps -e drugforge-data
RUN python -m pip install --no-deps -e drugforge-dataviz
RUN python -m pip install --no-deps -e drugforge-docking
RUN python -m pip install --no-deps -e drugforge-ml
RUN python -m pip install --no-deps -e drugforge-modeling
RUN python -m pip install --no-deps -e drugforge-simulation

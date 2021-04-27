ARG TORCHSERVE_TAG=latest
FROM pytorch/torchserve:${TORCHSERVE_TAG}

COPY requirements.txt .
USER root
RUN apt-get update -y
RUN apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev git
RUN pip install wheel && pip install -r requirements.txt
USER model-server

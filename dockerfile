# The ARG instruction defines a variable that users can pass at build-time
# to the builder with the docker build command using the --build-arg <varname>=<value> flag.
# If not specified, the default is nvidia/cuda:11.4.3-runtime-ubuntu20.04
ARG BASE_IMAGE=nvidia/cuda:11.4.3-runtime-ubuntu20.04

# The second ARG is likely intended to be a different name or used differently.
# If it's intended to override the base image, it should not be redeclared. 
# If it's intended for use in a later stage, it should have a different name.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
FROM ${PYTORCH_IMAGE} as dev-base

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash

# Install system dependencies
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    git \
    bash \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Upgrade pip
RUN /usr/bin/python3 -m pip install --upgrade pip

# Add the start.sh script to the root of the container
ADD start.sh /

# Make the start.sh script executable
RUN chmod +x /start.sh

# Define the default command to run when the container starts
CMD [ "/start.sh" ]

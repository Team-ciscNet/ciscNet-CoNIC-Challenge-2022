# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Edit the base image here, e.g., to use 
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/) 
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime AS cisc-net-base
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime AS cisc-net-base
# FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04 AS cisc-net-base
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 openslide-tools -y
# >>>>>>>>>>>>>>>>>>>>>>>>>
# ! DO NOT MODIFY
# <<<<<<<<<<<<<<<<<<<<<<<<<
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
USER algorithm
WORKDIR /opt/algorithm
ENV PATH="/home/algorithm/.local/bin:${PATH}"
RUN python -m pip install --user -U pip
# >>>>>>>>>>>>>>>>>>>>>>>>>
# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# >>>>>>>>>>>>>>>>>>>>>>>>>
# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# Copy all required files such that they are available within the docker image (code, weights, ...)
COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm source/ /opt/algorithm/source/
COPY --chown=algorithm:algorithm data/ /opt/algorithm/data/
COPY --chown=algorithm:algorithm model.pth /opt/algorithm/
COPY --chown=algorithm:algorithm model.json /opt/algorithm/

# Install required python packages via pip - you may adapt the requirements.txt to your needs
RUN python -m pip install --user -r requirements.txt


FROM cisc-net-base AS cisc-net-app
WORKDIR /opt/algorithm
# CMD ["conda", "run", "--no-capture-output", "-n", "conic_challenge_ve", "python", "-u", "train.py"]

# Entrypoint to your python code - executes process.py as a script
ENTRYPOINT python -m process $0 $@
# >>>>>>>>>>>>>>>>>>>>>>>>>

## ALGORITHM LABELS ##

# ! USER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=CISCNet
# >>>>>>>>>>>>>>>>>>>>>>>>>

# ! DO NOT MODIFY - ORGANIZER SPECIFIC
# <<<<<<<<<<<<<<<<<<<<<<<<<
# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=4
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=()
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=16G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=11.3
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=12G
# >>>>>>>>>>>>>>>>>>>>>>>>>

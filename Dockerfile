# syntax = docker/dockerfile:experimental
#
# By default it builds image for CPU.
# DOCKER_BUILDKIT=1 docker build -t pytorch_kfp_components:latest .
# Command for GPU/cuda - 
# DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime -t pytorch_kfp_components:latest .

ARG BASE_IMAGE=pytorch/pytorch:latest

FROM ${BASE_IMAGE}

COPY requirements.txt requirements.txt

RUN apt-get update

RUN apt-get install -y git

RUN git clone -b main https://github.com/chauhang/pytorch-pipeline

RUN pip3 install -r requirements.txt

ENV PYTHONPATH /workspace/pytorch-pipeline

WORKDIR /workspace/pytorch-pipeline

ENTRYPOINT /bin/bash

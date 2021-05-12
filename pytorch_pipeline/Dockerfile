FROM pytorch/pytorch:latest

COPY requirements.txt requirements.txt

RUN apt-get update

RUN apt-get install -y git

RUN git clone -b main https://github.com/chauhang/pytorch-pipeline

RUN pip3 install -r requirements.txt

ENV PYTHONPATH /workspace/pytorch-pipeline

WORKDIR /workspace/pytorch-pipeline

ENTRYPOINT /bin/bash

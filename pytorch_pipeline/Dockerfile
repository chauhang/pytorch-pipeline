FROM pytorch/pytorch:latest

COPY requirements.txt requirements.txt

RUN apt-get update

RUN apt-get install -y git

RUN git clone -b fix/runpipeline_notebook https://github.com/jagadeeshi2i/pytorch-pipeline

RUN pip3 install -r requirements.txt

ENV PYTHONPATH /workspace/pytorch-pipeline

WORKDIR /workspace/pytorch-pipeline

ENTRYPOINT /bin/bash

FROM gcr.io/deeplearning-platform-release/pytorch-gpu:latest

RUN pip install image pandas sklearn torch torchtext

ENV PYTHONPATH /component

COPY ./src /component/src

COPY requirements.txt .

RUN pip3 install -r requirements.txt

WORKDIR /component

ENTRYPOINT /bin/bash

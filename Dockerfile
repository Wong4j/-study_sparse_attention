FROM nvcr.io/nvidia/pytorch:21.05-py3
RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
       sudo \
       libaio-dev \	
 && rm -rf /var/lib/apt/lists/*

RUN pip install deepspeed sklearn tensorboardX boto3 h5py
RUN pip install -U --pre triton

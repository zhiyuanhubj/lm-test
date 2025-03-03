#!/bin/bash
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"


conda activate vllm


if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi


MODEL_PATH=$1
NGPU=$2

# convert to int and compute NGPU-1
NGPU=${NGPU:-1}
NUM=$((NGPU - 1))


for i in $(seq 0 $NUM)
do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization=0.9 \
        --max-num-seqs=200 \
        --host 127.0.0.1 --tensor-parallel-size 1 \
        --port $((8000+i)) \
        --max-model-len 32000 \
    &
done
#!/bin/bash

WORK_PATH='/root/mount/gozaru'

modelPath='/root/mount/models/ELYZA-japanese-Llama-2-7b-fast-instruct-q2_K.gguf'
inputPath=${WORK_PATH}/output/lora-LATEST.gguf
outputPath=${WORK_PATH}/output/lora-ITERATION.gguf
loraOutputPath=${WORK_PATH}/loraout/lora-ITERATION.bin
trainPath=${WORK_PATH}/input.txt

/root/llama.cpp/finetune \
        --model-base ${modelPath} \
        --checkpoint-in  ${inputPath} \
        --checkpoint-out ${outputPath} \
        --lora-out ${loraOutputPath} \
        --train-data ${trainPath} \
        --save-every 10 \
        --threads 6 \
        --adam-iter 60 \
        --batch 5 \
        --ctx 64 \
        --sample-start '[INST]' \
        --use-checkpointing
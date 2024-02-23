#!/bin/bash

WORK_PATH='/root/mount/gozaru'

modelPath='/root/mount/models/ELYZA-japanese-Llama-2-7b-fast-instruct-q2_K.gguf'
loraPath=${WORK_PATH}/loraout/lora-LATEST.bin

prompt='[INST]太陽が沈むとどうなるのか？[INST]'

/root/llama.cpp/main \
        --model ${modelPath} \
        --lora ${loraPath} \
        --n-predict 256 \
        --prompt ${prompt}

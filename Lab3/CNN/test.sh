#!/bin/bash

models=("lenet" "alexnet" "resnet18" "googlenet" "vgg11" "efficientnet" "mnasnet" "mobilenetv3" "shufflenetv2" "squeezenet")
schedulers=("plateau" "cosine")

for model in "${models[@]}"; do
    for scheduler in "${schedulers[@]}"; do
        echo "Testing model: $model, lr_scheduler: $scheduler"
        python3 main.py --model "$model" --lr_scheduler "$scheduler"
    done
done

for model in "${models[@]}"; do
    echo "Feature analysis: $model"
    python3 feature_analysis.py --model "$model"
done
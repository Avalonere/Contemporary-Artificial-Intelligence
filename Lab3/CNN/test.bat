@echo off
setlocal enabledelayedexpansion

set models=lenet alexnet resnet18 googlenet vgg11 efficientnet mnasnet mobilenetv3 shufflenetv2 squeezenet
set schedulers=plateau cosine

for %%m in (%models%) do (
    for %%s in (%schedulers%) do (
        echo Testing model: %%m, lr_scheduler: %%s
        python main.py --model %%m --lr_scheduler %%s
    )
)

for %%m in (%models%) do (
    echo Feature Analysis: %%m
    python feature_analysis.py --model %%m
)
@echo off
setlocal enabledelayedexpansion

:: BART实验
for %%s in (2333, 4001, 6007, 8009, 9001) do (
    echo Running BART with seed %%s
    python main.py --model bart --lr 5e-5 --batch_size 8 --seed %%s
)

:: T5实验
for %%s in (2333, 4001, 6007, 8009, 9001) do (
    echo Running T5 with seed %%s
    python main.py --model t5 --lr 1e-3 --batch_size 16 --seed %%s
)

echo All experiments completed
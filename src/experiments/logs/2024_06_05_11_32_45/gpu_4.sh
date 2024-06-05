#!/bin/bash
source ~/.virtualenvs/nca-alife/bin/activate
cd ~/nca-alife/src
export CUDA_VISIBLE_DEVICES=4

touch ./experiments/logs/2024_06_05_11_32_45/job_000.status
python main.py --lr=0.01  --bs=32 --model="transformer" --save_dir="~/data/transformer_0.01_32"  &> ./experiments/logs/2024_06_05_11_32_45/job_000.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_000.status

touch ./experiments/logs/2024_06_05_11_32_45/job_004.status
python main.py --lr=0.001 --bs=32 --model="transformer" --save_dir="~/data/transformer_0.001_32" &> ./experiments/logs/2024_06_05_11_32_45/job_004.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_004.status


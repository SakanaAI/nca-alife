#!/bin/bash
source ~/.virtualenvs/nca-alife/bin/activate
cd ~/nca-alife/src
export CUDA_VISIBLE_DEVICES=6

touch ./experiments/logs/2024_06_05_11_32_45/job_002.status
python main.py --lr=0.01  --bs=64 --model="transformer" --save_dir="~/data/transformer_0.01_64"  &> ./experiments/logs/2024_06_05_11_32_45/job_002.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_002.status

touch ./experiments/logs/2024_06_05_11_32_45/job_006.status
python main.py --lr=0.001 --bs=64 --model="transformer" --save_dir="~/data/transformer_0.001_64" &> ./experiments/logs/2024_06_05_11_32_45/job_006.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_006.status


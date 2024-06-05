#!/bin/bash
source ~/.virtualenvs/nca-alife/bin/activate
cd ~/nca-alife/src
export CUDA_VISIBLE_DEVICES=7

touch ./experiments/logs/2024_06_05_11_32_45/job_003.status
python main.py --lr=0.01  --bs=64 --model="cnn"         --save_dir="~/data/cnn_0.01_64"          &> ./experiments/logs/2024_06_05_11_32_45/job_003.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_003.status

touch ./experiments/logs/2024_06_05_11_32_45/job_007.status
python main.py --lr=0.001 --bs=64 --model="cnn"         --save_dir="~/data/cnn_0.001_64"         &> ./experiments/logs/2024_06_05_11_32_45/job_007.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_007.status


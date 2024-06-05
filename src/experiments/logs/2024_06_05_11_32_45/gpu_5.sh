#!/bin/bash
source ~/.virtualenvs/nca-alife/bin/activate
cd ~/nca-alife/src
export CUDA_VISIBLE_DEVICES=5

touch ./experiments/logs/2024_06_05_11_32_45/job_001.status
python main.py --lr=0.01  --bs=32 --model="cnn"         --save_dir="~/data/cnn_0.01_32"          &> ./experiments/logs/2024_06_05_11_32_45/job_001.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_001.status

touch ./experiments/logs/2024_06_05_11_32_45/job_005.status
python main.py --lr=0.001 --bs=32 --model="cnn"         --save_dir="~/data/cnn_0.001_32"         &> ./experiments/logs/2024_06_05_11_32_45/job_005.out
echo $? > ./experiments/logs/2024_06_05_11_32_45/job_005.status


#!/bin/bash

#SBATCH --job-name=train_net
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH -o train_net_medium_new_annotations.log

eval "$(conda shell.bash hook)"
conda activate DL4NLP_yolo

python3 yolo/yolo.py --model yolo11m.pt --device 1 --epochs 1000
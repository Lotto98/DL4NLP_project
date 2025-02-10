#!/bin/bash

#SBATCH --job-name=train_net
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH -o train_net_nano_new_annotations_1216_resumed.log

eval "$(conda shell.bash hook)"
conda activate DL4NLP_yolo

python3 yolo/yolo_training.py --model resume --epochs 1000 --batch 24 --path-model runs/detect/train13/weights/last.pt
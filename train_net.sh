#!/bin/bash

#SBATCH --job-name=train_net
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH -o train_net_medium.log

eval "$(conda shell.bash hook)"
conda activate DL4NLP_yolo


#python3 train_net.py --num-gpus 1 --config-file configs/diffdet.ami.ast.yaml
python3 yolo.py --model yolo11n.pt --device 2 --epochs 500
python3 yolo.py --model yolo11m.pt --device 2 --epochs 500
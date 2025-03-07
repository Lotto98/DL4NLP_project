#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o prepare_dataset.log

eval "$(conda shell.bash hook)"
conda activate DL4NLP_yolo

python3 dataset_creation_AMI.py --split all
python3 yolo/yolo_dataset.py --split all
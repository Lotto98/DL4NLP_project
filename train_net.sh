#!/bin/bash

#SBATCH --job-name=train_net
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH -o train_net.log

#importante: crea il tuo env conda (lo puoi fare con i comandi standard)
#poi attivalo in questo modo:
eval "$(conda shell.bash hook)"
conda activate DL4NLP

#esegui il tuo codice qua
python3 train_net.py --num-gpus 1 --config-file configs/diffdet.ami.ast.yaml
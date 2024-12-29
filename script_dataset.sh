#!/bin/bash

#specifica che cosa servirà al tuo programma:
#in questo caso:
# -nome job
# -tempo massimo di esecuzione
# -numero task da eseguire (comandi)
# -cpu da assegnare a ogni comando (puoi specificare anche GPU con un comando simile)
# -specifica il file di log per debug e per vedere a che punto è la tua eseguzione

#SBATCH --job-name=prepare_data
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o prepare_dataset.log

#importante: crea il tuo env conda (lo puoi fare con i comandi standard)
#poi attivalo in questo modo:
eval "$(conda shell.bash hook)"
conda activate DL4NLP

#esegui il tuo codice qua
python3 dataset_creation_AMI.py --split all

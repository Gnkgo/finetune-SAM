#!/bin/bash

#SBATCH -n 2
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --tmp=4000
#SBATCH --job-name=finetune
#SBATCH --output=finetune.out
#SBATCH --error=finetune.err
#SBATCH --gpus=1

bash /cluster/home/jbrodbec/finetune-SAM/val_singlegpu_demo.sh





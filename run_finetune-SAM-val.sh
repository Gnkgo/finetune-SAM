#!/bin/bash

#SBATCH -n 2
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --tmp=4000
#SBATCH --job-name=finetune-val
#SBATCH --output=finetune-val.out
#SBATCH --error=finetune-val.err
#SBATCH --gpus=1

bash /cluster/home/jbrodbec/finetune-SAM/val_singlegpu_demo.sh

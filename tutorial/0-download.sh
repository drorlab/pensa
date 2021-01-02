#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=rondror
#SBATCH --constraint=GPU_MEM:12GB
#SBATCH --qos=high_p

python ~/pensa/scripts/get_tutorial_datasets.py -d "./mor-data"


#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="SSLHZH"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL


ckptPath=$1
threshold=$2
range=$3

python src/train/test.py MultNat "$ckptPath" --gpus 0 --classification_threshold "$threshold" --levy_holt --range "$range"

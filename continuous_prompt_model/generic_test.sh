#!/bin/bash
#SBATCH --partition=PGR-Standard
#SBATCH --job-name="SSLHZH"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=tianyi.li@ed.ac.uk
#SBATCH --mail-type=ALL

n_list=(100 100 100 100 100 100 100 100 100 100 75 75 75 75 75 75 75 75 75 75 50 50 50 50 50 50 50 50 50 50 25 25 25 25 25 25 25 25 25 25 10 10 10 10 10 10 10 10 10 10 5 5 5 5 5 5 5 5 5 5 1 1 1 1 1 1 1 1 1 1)
k_list=(10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1 10 9 8 7 6 5 4 3 2 1)

ckptPath=$1
threshold=$2
nk_id=$3
range=$4
num_patts=${n_list["$nk_id"]}
num_toks_per_patt=${k_list["$nk_id"]}

python src/train/test.py "$ckptPath" --classification_threshold "$threshold" --use_antipatterns --levy_holt --num_patterns "$num_patts" --num_tokens_per_pattern "$num_toks_per_patt" --range "$range"

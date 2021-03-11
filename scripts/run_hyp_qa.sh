#!/bin/bash
#
#SBATCH --job-name=hyp_distil_qa
#SBATCH --output=./output/hyp_qa_distilbart.txt
#SBATCH --gres=gpu:1
#SBATCH --partition=2080ti-long
#SBATCH --time=0-24:00:00
#SBATCH --mem=16GB

wandb agent haroldrubio/autoregkd/a3ciz8q1
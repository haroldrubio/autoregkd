#!/usr/bin/env bash
#
#SBATCH --partition=rtx8000-long
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=3

srun scripts/run_qa.sh
#!/usr/bin/env bash
#
#SBATCH --partition=rtx8000-long
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=12

srun scripts/run_seq2seq.sh
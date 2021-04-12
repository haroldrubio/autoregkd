#!/usr/bin/env bash
#
#SBATCH --partition=rtx8000-long
#SBATCH --gres=gpu:2
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=12
#SBATCH --output=output/interpolation_xsum_12_3_run_1.log

srun scripts/run_seq2seq.sh
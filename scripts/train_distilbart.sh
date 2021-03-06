#!/usr/bin/env bash

mkdir output

sbatch --partition=2080ti-long --gres=gpu:4 --mem=16384 --output=output/distilbart_xsum.log scripts/distilbart.sh
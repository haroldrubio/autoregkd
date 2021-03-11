#!/usr/bin/env bash

mkdir output

sbatch --partition=rtx8000-long --gres=gpu:2 --mem=16384 --output=output/distilbart_xsum.log scripts/distilbart.sh
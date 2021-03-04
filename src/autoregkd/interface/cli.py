import sys, os
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)

from ..utils.custom_args import ModelArguments, DatasetArguments

def experiment(**config):
    """Train a BART model"""
    from ..training.train import training
    
    parser = HfArgumentParser((ModelArguments, DatasetArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training(model_args, data_args, training_args)

import sys, os
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)

from ..utils.custom_args import ModelArguments, DatasetArguments, DistilArguments

def experiment(**config):
    """Train a BART model"""
    from ..training.train import training
    
    parser = HfArgumentParser((ModelArguments, DatasetArguments, DistilArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Parse indices arguments
    # encoder_layer_indices
    if type(model_args.encoder_layer_indices) == str:
        e_indices = model_args.encoder_layer_indices.split(',')
        d_indices = model_args.decoder_layer_indices.split(',')

        e_indices = [int(ele) for ele in e_indices]
        d_indices = [int(ele) for ele in d_indices]

        model_args.encoder_layer_indices = e_indices
        model_args.decoder_layer_indices = d_indices

    training(model_args, data_args, training_args)

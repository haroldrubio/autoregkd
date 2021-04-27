import sys, os
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)

from ..utils.custom_args import ModelArguments, DataTrainingArguments, DistilArguments

def experiment(**config):
    """Train a BART model"""
    from ...hf_qa.run_qa import main
    from ...hf_sum.run_summarization import main as main_sum
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, DistilArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'squad' in data_args.dataset_name:
        main(model_args, data_args, training_args)
    else:
        main_sum(model_args, data_args, training_args)    

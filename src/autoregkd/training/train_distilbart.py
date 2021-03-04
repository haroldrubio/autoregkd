"""
Training script for DistilBART
Based on https://github.com/huggingface/transformers/blob/master/examples/seq2seq/run_seq2seq.py
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import nltk
import numpy as np
from filelock import FileLock

import torch
import transformers
from transformers import (
    BartTokenizer,
    BartModel,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed
)

from ..models.distilbart.configuration_distilbart import DistilBartConfig
from ..models.distilbart.modeling_distilbart import DistilBart

from datasets import load_dataset, load_metric


with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for model
    """
    model_name: str = field(
        default="facebook/bart-large",
        metadata={"help": "Name of BART model we will copy and fine-tune from (https://huggingface.co/models)"}
    )

    tokenizer_name: str = field(
        default="facebook/bart-large",
        metadata={"help": "Name of pre-trained BART tokenizer"}
    )

    encoder_layer_indices: Tuple = field(
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's encoder"}
    )

    decoder_layer_indices: Tuple = field(
        default=(0, 6, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's decoder"}
    )


@dataclass
class DatasetArguments:
    """
    Arguments for dataset
    """
    task: str = field(
        default="summarization",
        metadata={
            "help": "Name of the task, should be either summarization, question-answering, or dialogue-generation"}
    )

    dataset_name: str = field(
        default="xsum",
        metadata={"help": "Name of the dataset to use (https://huggingface.co/datasets). "
                          "Support xsum, squad, or conv_ai_2"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum number of input tokens"},
    )

    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum number of output tokens"},
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad to global max length or batch max length"}
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams used in beam search during evaluation and prediction steps "},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for replicability
    set_seed(training_args.seed)

    # Load dataset
    datasets = load_dataset(data_args.dataset_name)
    train_dataset = datasets['train']
    val_dataset = datasets['evaluation'] if 'evaluation' in datasets.keys() else None
    test_dataset = datasets['test'] if 'test' in datasets.keys() else None

    # Do train-test split for ConvAI2 since there's no validation split
    if not val_dataset:
        pass

    # Get column names
    column_names = train_dataset.column_names
    if val_dataset:
        assert column_names == val_dataset.column_names
    if test_dataset:
        assert column_names == test_dataset.column_names

    # DistilBART configuration
    config = DistilBartConfig(
        encoder_layer_indices=list(model_args.encoder_layer_indices),
        decoder_layer_indices=list(model_args.decoder_layer_indices)
    )

    # BART tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)

    # DistilBART model
    bart_model = BartModel.from_pretrained(model_args.model_name)
    distilbart_model = DistilBart(config=config, bart_model=bart_model)

    # Max lengths
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_xsum(examples):
        inputs = examples["document"]
        targets = examples["summary"]

        # Tokenize source
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize target
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if train_dataset:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_xsum,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    if val_dataset:
        if data_args.max_val_samples is not None:
            val_dataset = val_dataset.select(range(data_args.max_val_samples))
        val_dataset = val_dataset.map(
            preprocess_xsum,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    if test_dataset:
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_xsum,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )

    if data_args.task == "summarization":
        metric_name = "rouge"
    elif data_args.task == "question-answering":
        metric_name = "f1"
    elif data_args.task == "ialogue-generation":
        metric_name = "f1"
    else:
        raise ValueError("Unsupported task")

    metric = load_metric(metric_name)

    def postprocess_text(preds, labels):
        if data_args.task == "summarization":
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Preprocess
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = {}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {key: round(value, 4) for key, value in result.items()}

        return result

    # Trainer
    trainer = Seq2SeqTrainer(
        model=distilbart_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Training
    logging("*** Training ***")
    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logging.info("*** Evaluating ***")
    results = {}
    metrics = trainer.evaluate(max_length=max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval")
    max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(val_dataset)
    metrics["eval_samples"] = min(max_val_samples, len(val_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return results


if __name__ == '__main__':
    main()

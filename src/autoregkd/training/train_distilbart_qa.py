"""
Training script to fine-tune DistilBART for the Question-Answering task
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
import torch.nn as nn

import transformers
from transformers import (
    BartConfig,
    BartTokenizer,
    BartModel,
    BartForQuestionAnswering,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed
)

from ..models.distilbart.configuration_distilbart import DistilBartConfig
from ..models.distilbart.modeling_distilbart import (
    create_new_student,
    copy_to_student
)

from datasets import load_dataset, load_metric


with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments for model
    """
    use_hf_model: bool = field(
        default=False,
        metadata={"help": "Whether to use Huggingface's BART model or custom BART model"}
    )

    model_name: str = field(
        default="facebook/bart-large-xsum",
        metadata={"help": "Name of BART model we will copy and fine-tune from (https://huggingface.co/models)"}
    )

    tokenizer_name: str = field(
        default="facebook/bart-large-xsum",
        metadata={"help": "Name of pre-trained BART tokenizer"}
    )

    encoder_layer_indices: Tuple = field(
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's encoder"}
    )

    decoder_layer_indices: Tuple = field(
        default=(0, 1, 2),
        metadata={"help": "Indices of layers to copy from the teacher model's decoder"}
    )


@dataclass
class DatasetArguments:
    """
    Arguments for dataset
    """
    task: str = field(
        default="question-answering",
        metadata={
            "help": "Name of the task. Support only question-answering at the moment"}
    )

    dataset_name: str = field(
        default="squad",
        metadata={"help": "Name of the dataset to use (https://huggingface.co/datasets). "
                          "Support squad and squadv2"}
    )

    use_v2: bool = field(
        default=False,
        metadata={"help": "Whether to use SQuADv2 or not. Default to false (use SQuADv1)"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing"},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total sequence length for source text after tokenization"},
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to pad to global max length or batch max length"}
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not"},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set"
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

    def __post_init__(self):
        # Update dataset name if we want to use v2 (for SQuAD)
        if self.dataset_name == "squad" and self.use_v2:
            self.dataset_name = "squad_v2"


def main():
    logging.info("GPUs available: {}. Number of GPUs: {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
    torch.cuda.empty_cache()

    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Enable mixed precision training on CUDA device(s)
    if torch.cuda.is_available() and not training_args.no_cuda:
        training_args.fp16 = True
        logging.info("Mixed precision training enabled.")

    # Set seed for replicability
    set_seed(training_args.seed)

    # Load dataset
    datasets = load_dataset(data_args.dataset_name)

    train_dataset, eval_dataset, test_dataset = None, None, None
    if training_args.do_train:
        train_dataset = datasets['train']
    if training_args.do_eval:
        eval_dataset = datasets['validation'] if 'validation' in datasets.keys() else None
    if training_args.do_predict:
        test_dataset = datasets['test'] if 'test' in datasets.keys() else None

    # Get column names
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # BART tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)

    def freeze_weights(module: nn.Module):
        """
        Freeze the weights of a module to accelerate training
        """
        for param in module.parameters():
            param.requires_grad = False

    if model_args.use_hf_model:
        # Load a pre-trained checkpoint for BART
        config = BartConfig()
        student_model = BartForQuestionAnswering(config=config).from_pretrained(model_args.model_name).eval()

    else:
        # Create a DistilBart model with layers copied from the original BART model
        # BART teacher model
        teacher_model = BartForQuestionAnswering.from_pretrained(model_args.model_name).eval()

        # Extract the teacher's configuration
        teacher_config = teacher_model.config.to_diff_dict()
        teacher_config.update({
            "encoder_layers": len(list(model_args.encoder_layer_indices)),
            "decoder_layers": len(list(model_args.decoder_layer_indices))
        })

        # DistilBART configuration
        student_config = DistilBartConfig(
            encoder_layer_indices=list(model_args.encoder_layer_indices),
            decoder_layer_indices=list(model_args.decoder_layer_indices),
            **teacher_config
        )

        # DistilBART model
        student_model = create_new_student(teacher_model=teacher_model, config=student_config).eval()

        # Copy the weights
        copy_to_student(teacher_model=teacher_model,
                        student_model=student_model,
                        config=student_config)

        # Freeze the encoder's parameters
        freeze_weights(student_model.model.encoder)

        # Freeze the decoder's positional and token embeddings
        freeze_weights(student_model.model.decoder.embed_tokens)
        freeze_weights(student_model.model.decoder.embed_positions)

        encoder_trainable_params = sum(p.numel() for p in student_model.model.encoder.parameters() if p.requires_grad)
        assert(
            encoder_trainable_params == 0
        ), "Expected the student's encoder to be frozen. Got {} trainable parameters".format(encoder_trainable_params)

    # Max lengths
    max_source_length = data_args.max_source_length
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_xsum(examples):
        """
        Pre-process examples from XSum
        :param examples:
        :return:
        """
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

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_xsum,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length

        if not eval_dataset:
            raise ValueError("No eval dataset available.")
            return

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_xsum,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length

        if not test_dataset:
            raise ValueError("No test dataset available.")
            return

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
            model=student_model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )

    if data_args.task == "summarization":
        metric_name = "rouge"
    else:
        raise ValueError("Unsupported task.")

    metric = load_metric(metric_name)

    def postprocess_text(preds, labels):
        if data_args.task == "summarization":
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        else:
            raise ValueError("Unsupported task.")

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            raise ValueError("Unsupported metric.")

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Eval steps (should be 4 times per epoch)
    if training_args.do_train:
        training_args.eval_steps = max(round(len(train_dataset) / training_args.train_batch_size / 4.), 1)
        training_args.logging_steps = training_args.eval_steps
        training_args.save_steps = training_args.eval_steps

    # Trainer
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    # Early-stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=4)
    trainer.add_callback(early_stopping)

    # Training
    if training_args.do_train:
        logging.info("*** Training ***")
        train_result = trainer.train()
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
    results = {}
    if training_args.do_eval:
        logging.info("*** Evaluating ***")

        metrics = trainer.evaluate(
            max_length=max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="eval")

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logging.info("*** Testing ***")

        test_results = trainer.predict(
            test_dataset,
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="test"
        )

        metrics = test_results.metrics
        print(metrics)
        max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
        metrics["test_samples"] = min(max_test_samples, len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        """
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                test_preds = [pred.strip() for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))
        """

    return results


if __name__ == '__main__':
    main()
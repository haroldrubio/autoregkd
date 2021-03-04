# This class is for being able to pass additional arguments to the trainer
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple
from transformers import TrainingArguments

@dataclass
class CustomArguments(TrainingArguments):
    test_name: str = field(
        default="No World",
        metadata={
            "help": (
                "A simple test"
            )
        },
    )

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
        default=(0, 1, 2, 3, 4, 5),
        metadata={"help": "Indices of layers to copy from the teacher model's encoder"}
    )

    decoder_layer_indices: Tuple = field(
        default=(0, 2, 5),
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
    # ==== QA ARGS ====
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "FOR QA: The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "FOR QA: If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "FOR QA: The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "FOR QA: When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "FOR QA: The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "FOR QA: The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
# This class is for being able to pass additional arguments to the trainer
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Tuple
from transformers import Seq2SeqTrainingArguments

@dataclass
class DistilArguments(Seq2SeqTrainingArguments):
    test_type: str = field(
        default="distilbart",
        metadata={"help": "The type of model"
                          "Supports 'distilbart' and 'interpolation'"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    perform_distillation: bool = field(
        default=True,
        metadata={
            "help": "Use a distilled model or not"
        },
    )
    loss_type: str = field(
        default="finetune",
        metadata={"help": "Supports finetune, interpolate"},
    )
    num_decoder_layers: int = field(
        default=3,
        metadata={"help": "Number of decoder layers to copy"}
    )
    # ----------Swap Prob Args----------
    swap_prob: float = field(
        default=0,
        metadata={"help": "When performing interpolation, set a constant swapping rate"}
    )
    max_prob: float = field(
        default=-1,
        metadata={"help": "When performing interpolation, set the maximum swapping probability"}
    )
    cool_down: float = field(
        default=-1,
        metadata={"help": "When performing interpolation, set the percentage of training time to be cooling down"}
    )
    conn_time: float = field(
        default=-1,
        metadata={"help": "When performing interpolationv2s, shrink the cooldown intervals by this percentage"}
    )
    interpolation_period: float = field(
        default=-1,
        metadata={"help": "When performing interpolationv2s, decide the PERCENTAGE of training time to interpolate over"}
    )
    plad: float = field(
        default=-1,
        metadata={"help": "When performing interpolationv2s, decide the per-level annealing duration as a PERCENTAGE of IP"}
    )
    reverse_probs: bool = field(
        default=False,
        metadata={
            "help": "Reverses the order of annealing"
        },
    )
    # ----------Swap Prob Args----------
    enc_interpolate: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform interpolation on the encoder"
        },
    )
    dec_interpolate: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform interpolation on the encoder"
        },
    )
    dec_interpolate_type: str = field(
        default="interpolate",
        metadata={"help": "Supports interpolate, interpolatev2s"},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": "Keep the dataset in memory instead of writing it to a cache file"
        },
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
    val_on_training: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, evaluate on the training set"
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

"""
Training script to fine-tune DistilBART for the Question-Answering task
Based on https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb#scrollTo=jwZn78Nfn1Sl
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple
from collections import defaultdict, OrderedDict

import nltk
import numpy as np
from tqdm.auto import tqdm
from filelock import FileLock

import torch
import torch.nn as nn

import transformers
from transformers import (
    BartConfig,
    BartTokenizerFast,
    BartForQuestionAnswering,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from ..models.distilbart.configuration_distilbart import DistilBartConfig
from ..models.distilbart.modeling_distilbart import (
    create_new_student,
    copy_to_student
)
from ..models.interpolation.modeling_interpolation import InterpolationBartForQuestionAnswering

from .trainer_qa import QuestionAnsweringTrainer, QuestionAnsweringKDTrainer, QuestionAnsweringInterpolationTrainer
from .utils import AddAtEpochCallback
from .utils_qa import postprocess_qa_predictions

from datasets import load_dataset, load_metric

with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def default_logdir(use_v2: bool) -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    version = "v2" if use_v2 else "v1"
    return os.path.join("runs", "squad_{}".format(version), current_time + "_" + socket.gethostname())


@dataclass
class ModelArguments:
    """
    Arguments for model
    """
    model_type: str = field(
        default="interpolation",
        metadata={"help": "Which type of model to use. Can choose among huggingface (HF's model for checkpoint "
                          "evaluation), distilbart (Distilbart paper replication), "
                          "and interpolation (Interpolation model)"}
    )

    model_name: str = field(
        default="Primer/bart-squad2",
        metadata={"help": "Name of BART model we will copy and fine-tune from (https://huggingface.co/models)"}
    )

    tokenizer_name: str = field(
        default="Primer/bart-squad2",
        metadata={"help": "Name of pre-trained BART tokenizer"}
    )

    student_encoder_layer_indices: Tuple = field(
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's encoder"}
    )

    student_decoder_layer_indices: Tuple = field(
        default=(3, 7, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's decoder"}
    )

    freeze_encoder: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the encoder. Default to True"}
    )

    freeze_embedding: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the embeddings (including token embeddings and positional embeddings). "
                          "Default to True"}
    )

    freeze_qa_head: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to freeze the QA task-specific linear layer. Default to True"}
    )

    # ----------------------------------------- #
    # KD-specific (Distilbart) hyper-parameters #
    # ----------------------------------------- #
    use_kd_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add knowledge-distillation loss (logits and hidden loss). "
                          "If False, the loss only includes data cross-entropy loss (same as SFT approach)"}
    )

    alpha_data: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weight for data loss. Default to 1.0"}
    )

    alpha_logits: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight for logits loss. Default to 0.0 (does not contribute to the total loss)"}
    )

    alpha_hidden: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight for hidden state loss. Default to 0.0 (does not contribute to the total loss)"}
    )

    normalize_hidden: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize hidden states before computing the loss. "
                          "Only useful if use KD loss and alpha_hidden greater than 0"}
    )

    # --------------------------------------- #
    # Interpolation-specific hyper-parameters #
    # --------------------------------------- #
    interpolation_type: Optional[str] = field(
        default="stochastic",
        metadata={"help": "Type of interpolation. Must be either stochastic or linear"}
    )

    num_interpolation_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "Number of interpolation epochs. Must be at most the total number of training epochs."
                          "Default to 5"}
    )

    learnable_p: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to make p learnable. If set to True, interpolation scheduler becomes unncessary"}
    )

    alpha_p: Optional[float] = field(
        default=None,
        metadata={"help": "Regularization factor to encourage p to be high (close to 1)"}
    )

    interpolation_p: Optional[float] = field(
        default=0.0,
        metadata={"help": "Starting probability for interpolation modules. Default to 0.0"}
    )

    max_prob: Optional[float] = field(
        default=1.0,
        metadata={"help": "Maximum possible probability for interpolation modules. Default to 1.0"}
    )

    per_level_annealing_duration: Optional[float] = field(
        default=0.2,
        metadata={"help": "How long each layer's annealing duration is, measure in fraction of the number of "
                          "interpolation steps. Default to 0.2"}
    )

    step_size: Optional[int] = field(
        default=1,
        metadata={"help": "How often the scheduler should update (it update every `step_size` steps). "
                          "Default to 1"}
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

    max_seq_length: Optional[int] = field(
        default=384,
        metadata={"help": "The maximum total sequence length for source text after tokenization"},
    )

    doc_stride: int = field(
        default=128,
        metadata={"help": "The authorized overlap between parts of context when splitting is necessary"}
    )

    pad_to_max_length: bool = field(
        default=True,
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

    n_best_size: Optional[int] = field(
        default=20,
        metadata={"help": "Number of best start/end logits allowed"}
    )

    max_answer_length: Optional[int] = field(
        default=30,
        metadata={"help": "Maximum answer length allowed"}
    )

    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
                    "the score of the null answer minus this threshold, the null answer is selected for this example. "
                    "Only useful when `use_v2=True`."
        }
    )

    num_evals_per_epoch: Optional[int] = field(
        default=4,
        metadata={"help": "Number of evaluations per epoch. Default to 4"}
    )

    def __post_init__(self):
        # Update dataset name if we want to use v2 (for SQuAD)
        if self.dataset_name == "squad" and self.use_v2:
            self.dataset_name = "squad_v2"


def main():
    logger.info("GPUs available: {}. Number of GPUs: {}".format(torch.cuda.is_available(), torch.cuda.device_count()))
    torch.cuda.empty_cache()

    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Number of GPU(s)
    logger.info(
        "GPUs available: {}. Number of GPUs: {}".format(torch.cuda.is_available(), torch.cuda.device_count())
    )

    # Enable mixed precision training on CUDA devices
    if not torch.cuda.is_available() or training_args.no_cuda:
        training_args.fp16 = False
        logger.info("Mixed precision training disabled.")

    # Update output dir if necessary
    if data_args.use_v2:
        training_args.output_dir += "squadv2/"
        logger.info("Using SQuADv2.0")
    else:
        training_args.output_dir += "squadv1/"
        logger.info("Using SQuADv1.1")
    if model_args.model_type == "interpolation":
        training_args.output_dir += "minp_{}_maxp_{}_plad_{}_step_{}_seed_{}/".format(model_args.interpolation_p,
                                                                                      model_args.max_prob,
                                                                                      model_args.per_level_annealing_duration,
                                                                                      model_args.step_size,
                                                                                      training_args.seed)

    # Detect and get last checkpoint
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed for replicability
    set_seed(training_args.seed)

    # Set Tensorboard logger dir
    tensorboard_logdir = default_logdir(data_args.use_v2)
    training_args.logger_dir = tensorboard_logdir
    logger.info("Tensorboard logged to {}".format(tensorboard_logdir))

    # Load dataset
    datasets = load_dataset(data_args.dataset_name)

    train_dataset, eval_dataset, eval_examples, test_dataset = None, None, None, None
    if training_args.do_train:
        train_dataset = datasets["train"]
    if training_args.do_eval:
        eval_dataset = datasets["validation"] if "validation" in datasets.keys() else None
        # A copy of the eval dataset without preprocessing
        eval_examples = datasets["validation"] if "validation" in datasets.keys() else None
    if training_args.do_predict:
        test_dataset = datasets["test"] if "test" in datasets.keys() else None

    # Get column names
    if training_args.do_train:
        if train_dataset:
            column_names = train_dataset.column_names
        else:
            raise ValueError("No train dataset available. Please provide one to use --do_train.")
            return
    elif training_args.do_eval:
        if eval_dataset:
            column_names = eval_dataset.column_names
        else:
            raise ValueError("No eval dataset available. Please provide one to use --do_eval.")
            return
    elif training_args.do_predict:
        if test_dataset:
            column_names = test_dataset
        else:
            raise ValueError("No test dataset available. Please provide one to use --do_predict.")
            return
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # BART tokenizer
    tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    def freeze_weights(module: nn.Module):
        """
        Freeze the weights of a module to accelerate training
        """
        for param in module.parameters():
            param.requires_grad = False

    if model_args.model_type == "huggingface":
        # Load a pre-trained checkpoint for BART
        config = BartConfig().from_pretrained(model_args.model_name)
        teacher_model = None
        student_model = BartForQuestionAnswering(config=config).from_pretrained(model_args.model_name).eval()

    elif model_args.model_type == "distilbart":
        # Create a DistilBart model with layers copied from the original BART model
        # BART teacher model
        teacher_model = BartForQuestionAnswering.from_pretrained(model_args.model_name).eval()

        # Extract the teacher's configuration
        teacher_config = teacher_model.config.to_diff_dict()
        teacher_config.update({
            "encoder_layers": len(list(model_args.student_encoder_layer_indices)),
            "decoder_layers": len(list(model_args.student_decoder_layer_indices)),
        })

        # DistilBART configuration
        student_config = DistilBartConfig(
            student_encoder_layer_indices=list(model_args.student_encoder_layer_indices),
            student_decoder_layer_indices=list(model_args.student_decoder_layer_indices),
            student_encoder_layers=len(list(model_args.student_encoder_layer_indices)),
            student_decoder_layers=len(list(model_args.student_decoder_layer_indices)),
            **teacher_config
        )

        # DistilBART model
        student_model = create_new_student(teacher_model=teacher_model, config=student_config).eval()

        assert (
                len(student_model.model.encoder.layers) == len(list(model_args.student_encoder_layer_indices))
                and len(student_model.model.decoder.layers) == len(list(model_args.student_decoder_layer_indices))
        )

        # Copy the weights
        copy_to_student(teacher_model=teacher_model,
                        student_model=student_model,
                        config=student_config)

        # Freeze shared embeddings
        freeze_weights(student_model.model.shared)

        # Freeze the decoder's positional and token embeddings
        freeze_weights(student_model.model.encoder.embed_tokens)
        freeze_weights(student_model.model.encoder.embed_positions)
        freeze_weights(student_model.model.decoder.embed_tokens)
        freeze_weights(student_model.model.decoder.embed_positions)

        # Freeze the rest of encoder's parameters
        freeze_weights(student_model.model.get_encoder())

        encoder_trainable_params = sum(p.numel() for p in student_model.model.encoder.parameters() if p.requires_grad)
        assert (
                encoder_trainable_params == 0
        ), "Expected the student's encoder to be frozen. Got {} trainable parameters".format(encoder_trainable_params)

    elif model_args.model_type == "interpolation":
        if model_args.num_interpolation_epochs > training_args.num_train_epochs:
            model_args.num_interpolation_epochs = training_args.num_train_epochs
            logger.info("Number of interpolation epochs exceeds number of training epochs. "
                        "Setting number of interpolation epochs to number of training epochs")

        teacher_config = BartConfig().from_pretrained(model_args.model_name).to_diff_dict()
        student_config = DistilBartConfig(
            student_encoder_layer_indices=list(model_args.student_encoder_layer_indices),
            student_decoder_layer_indices=list(model_args.student_decoder_layer_indices),
            interpolation_type=model_args.interpolation_type,
            learnable_p=model_args.learnable_p,
            interpolation_p=model_args.interpolation_p,
            **teacher_config
        )

        teacher_model = None
        student_model = InterpolationBartForQuestionAnswering.from_pretrained(model_args.model_name, config=student_config)
        student_model.load_weights_to_student()
        student_model.freeze_weights(freeze_embedding=model_args.freeze_embedding,
                                     freeze_encoder=model_args.freeze_encoder,
                                     freeze_qa_head=model_args.freeze_qa_head)

        encoder_trainable_params = sum(p.numel() for p in student_model.model.encoder.parameters() if p.requires_grad)
        assert (
                encoder_trainable_params == 0
        ), "Expected the encoder to be frozen. Got {} trainable parameters".format(encoder_trainable_params)

        decoder_teacher_layers = [
            student_model.model.decoder.embed_tokens,
            student_model.model.decoder.embed_positions,
            student_model.model.decoder.layers,
            student_model.model.decoder.layernorm_embedding
        ]
        decoder_teacher_trainable_params = sum(
            sum(p.numel() for p in l.parameters() if p.requires_grad) for l in decoder_teacher_layers
        )
        assert (
                decoder_teacher_trainable_params == 0
        ), "Expected the teacher's decoder to be frozen. Got {} trainable parameters".format(
            decoder_teacher_trainable_params)

        if model_args.freeze_embedding:
            student_embedding_layers = [
                student_model.model.decoder.student_embed_tokens,
                student_model.model.decoder.student_embed_positions,
                student_model.model.decoder.student_layernorm_embedding
            ]
            decoder_student_trainable_embed_params = sum(
                sum(p.numel() for p in l.parameters() if p.requires_grad) for l in student_embedding_layers
            )
            assert (
                    decoder_student_trainable_embed_params == 0
            ), "Expected the student's decoder's embeddings to be frozen. Got {} trainable parameters" \
                .format(decoder_student_trainable_embed_params)

        if model_args.freeze_qa_head:
            qa_head_trainable_params = sum(p.numel() for p in student_model.qa_outputs.parameters() if p.requires_grad)
            assert (
                    qa_head_trainable_params == 0
            ), "Expected the QA head to be frozen. Got {} trainable parameters".format(qa_head_trainable_params)

    else:
        raise ValueError("Invalid model type - should be huggingface, distilbart, or interpolation")

    # Max lengths
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_squad_train(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if not train_dataset:
            raise ValueError("No train dataset available.")

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        train_dataset = train_dataset.map(
            preprocess_squad_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

    def preprocess_squad_eval(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        if not eval_dataset:
            raise ValueError("No eval dataset available.")
            return

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
            eval_examples = eval_examples.select(range(data_args.max_val_samples))

        eval_dataset = eval_dataset.map(
            preprocess_squad_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

        if data_args.max_val_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict:
        if not test_dataset:
            raise ValueError("No test dataset available.")
            return

        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

        test_dataset = test_dataset.map(
            preprocess_squad_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

        if data_args.max_test_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )

    # Load metric to use in evaluation
    if data_args.task == "question-answering":
        metric_name = "squad_v2" if data_args.use_v2 else "squad"  # EM/F1 scores
    else:
        raise ValueError("Unsupported task.")

    metric = load_metric(metric_name)

    def postprocess_squad(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.use_v2,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.use_v2:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(eval_preds):
        return metric.compute(predictions=eval_preds.predictions, references=eval_preds.label_ids)

    # Eval steps (should be ~4 times per epoch)
    if training_args.do_train:
        if training_args.do_eval:
            training_args.eval_steps = max(round(len(train_dataset) / training_args.train_batch_size / data_args.num_evals_per_epoch / training_args.gradient_accumulation_steps), 1)
            training_args.logger_steps = training_args.eval_steps
            training_args.save_steps = training_args.eval_steps
            logger.info("Evaluate every {} steps, or {} times per epoch".format(training_args.eval_steps,
                                                                                data_args.num_evals_per_epoch))
        else:
            training_args.evaluation_strategy = "no"

    # Trainer
    # Different trainers for different models
    if model_args.model_type == "huggingface":
        trainer = QuestionAnsweringTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocess_squad,
            compute_metrics=compute_metrics
        )
    elif model_args.model_type == "distilbart":
        trainer = QuestionAnsweringKDTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocess_squad,
            compute_metrics=compute_metrics,
            use_kd_loss=model_args.use_kd_loss,
            teacher_model=teacher_model,
            temperature=2.0,
            alpha_data=model_args.alpha_data,
            alpha_logits=model_args.alpha_logits,
            alpha_hidden=model_args.alpha_hidden
        )
    else:
        trainer = QuestionAnsweringInterpolationTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocess_squad,
            compute_metrics=compute_metrics,
            num_interpolation_epochs=model_args.num_interpolation_epochs,
            learnable_p=model_args.learnable_p,
            alpha_p=model_args.alpha_p,
            max_prob=model_args.max_prob,
            per_level_annealing_duration=model_args.per_level_annealing_duration,
            step_size=model_args.step_size
        )

    # Early-stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience=4)
    if model_args.model_type == "huggingface" or model_args.model_type == "distilbart":
        # Add directly to the trainer
        trainer.add_callback(early_stopping)
    else:
        # For interpolation models, we only add early stopping after the interpolation period is done
        callback = AddAtEpochCallback(trainer=trainer,
                                      num_interpolation_epochs=model_args.num_interpolation_epochs,
                                      callback=early_stopping)
        trainer.add_callback(callback)

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
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
        logger.info("*** Evaluating ***")

        # Make predictions and metrics
        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logger.info("*** Testing ***")

        # Raw predictions
        test_results = trainer.predict(test_dataset, metric_key_prefix="test")
        metrics = test_results.metrics

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

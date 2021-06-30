"""
Training script to fine-tune Seq2Seq DistilBART for tasks such as summarization
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
import torch.nn as nn

import transformers
from transformers import (
    BartConfig,
    BartTokenizer,
    BartModel,
    BartForConditionalGeneration,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from ..models.distilbart.configuration_distilbart import DistilBartConfig
from ..models.distilbart.modeling_distilbart import (
    create_new_student,
    copy_to_student
)
from ..models.interpolation.modeling_interpolation import InterpolationBartForConditionalGeneration

from .trainer_seq2seq import Seq2SeqTrainer, Seq2SeqKDTrainer, Seq2SeqInterpolationTrainer
from .utils import AddAtEpochCallback

from datasets import load_dataset, load_metric


with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", "xsum", current_time + "_" + socket.gethostname())


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
        default="facebook/bart-large-xsum",
        metadata={"help": "Name of BART model we will copy and fine-tune from (https://huggingface.co/models)"}
    )

    tokenizer_name: str = field(
        default="facebook/bart-large-xsum",
        metadata={"help": "Name of pre-trained BART tokenizer"}
    )

    student_encoder_layer_indices: Tuple = field(
        default=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        metadata={"help": "Indices of layers to copy from the teacher model's encoder"}
    )

    student_decoder_layer_indices: Tuple = field(
        default=(0, 6, 11),
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

    freeze_lm_head: Optional[bool] = field(
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
    num_interpolation_epochs: Optional[int] = field(
        default=5,
        metadata={"help": "Number of interpolation epochs. Must be at most the total number of training epochs."
                          "Default to 5"}
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
        default="summarization",
        metadata={
            "help": "Name of the task. Support only summarization at the moment"}
    )

    dataset_name: str = field(
        default="xsum",
        metadata={"help": "Name of the dataset to use (https://huggingface.co/datasets)"
                          "Support xsum"}
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
        default=1024,
        metadata={"help": "The maximum total sequence length for source text after tokenization"},
    )

    max_target_length: Optional[int] = field(
        default=256,
        metadata={"help": "The maximum total sequence length for target text after tokenization"},
    )

    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. "
                    "Default to max_target_length"
        },
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

    num_evals_per_epoch: Optional[int] = field(
        default=4,
        metadata={"help": "Number of evaluations per epoch. Default to 4"}
    )

    num_beams: Optional[int] = field(
        default=6,
        metadata={"help": "Number of beams used in beam search during evaluation and prediction steps "},
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    torch.cuda.empty_cache()

    parser = HfArgumentParser((ModelArguments, DatasetArguments, Seq2SeqTrainingArguments))
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
    logger.info("GPUs available: {}. Number of GPUs: {}".format(torch.cuda.is_available(), torch.cuda.device_count()))

    # Enable mixed precision training on CUDA devices
    if not torch.cuda.is_available() or training_args.no_cuda:
        training_args.fp16 = False
        logger.info("Mixed precision training disabled.")

    if model_args.model_type == "interpolation":
        training_args.output_dir += "minp_{}_maxp_{}_plad_{}_step_{}_seed_{}/".format(model_args.interpolation_p,
                                                                                      model_args.max_prob,
                                                                                      model_args.per_level_annealing_duration,
                                                                                      model_args.step_size,
                                                                                      training_args.seed)

    # Detect and get last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
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

    # Set Tensorboard logging dir
    tensorboard_logdir = default_logdir()
    training_args.logging_dir = tensorboard_logdir
    logger.info("Tensorboard logged to {}".format(tensorboard_logdir))

    # Load dataset
    datasets = load_dataset(data_args.dataset_name)

    train_dataset, eval_dataset, test_dataset = None, None, None
    if training_args.do_train:
        train_dataset = datasets["train"]
    if training_args.do_eval:
        eval_dataset = datasets["validation"] if "validation" in datasets.keys() else None
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

    # BART tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)

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
        student_model = BartForConditionalGeneration(config=config).from_pretrained(model_args.model_name).eval()

    elif model_args.model_type == "distilbart":
        # Create a DistilBart model with layers copied from the original BART model
        # BART teacher model
        teacher_model = BartForConditionalGeneration.from_pretrained(model_args.model_name).eval()

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
            interpolation_p=model_args.interpolation_p,
            **teacher_config
        )

        teacher_model = None
        student_model = InterpolationBartForConditionalGeneration(student_config)
        student_model.from_pretrained(model_args.model_name, config=student_config)
        student_model.load_weights_to_student()
        student_model.freeze_weights(freeze_embedding=model_args.freeze_embedding,
                                     freeze_encoder=model_args.freeze_encoder,
                                     freeze_lm_head=model_args.freeze_lm_head)

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
        ), "Expected the teacher's decoder to be frozen. Got {} trainable parameters".format(decoder_teacher_trainable_params)

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
            ), "Expected the student's decoder's embeddings to be frozen. Got {} trainable parameters"\
                .format(decoder_student_trainable_embed_params)

        if model_args.freeze_lm_head:
            lm_head_trainable_params = sum(p.numel() for p in student_model.lm_head.parameters() if p.requires_grad)
            assert (
                lm_head_trainable_params == 0
            ), "Expected the QA head to be frozen. Got {} trainable parameters".format(lm_head_trainable_params)

    else:
        raise ValueError("Invalid model type - should be huggingface, distilbart, or interpolation")

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
        if not train_dataset:
            raise ValueError("No train dataset available.")

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

        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    # Eval steps (should be 4 times per epoch)
    if training_args.do_train:
        if training_args.do_eval:
            training_args.eval_steps = max(round(len(train_dataset) / training_args.train_batch_size / data_args.num_evals_per_epoch / training_args.gradient_accumulation_steps), 1)
            training_args.logging_steps = training_args.eval_steps
            training_args.save_steps = training_args.eval_steps
            logger.info("Evaluate every {} steps, or {} times per epoch".format(training_args.eval_steps, data_args.num_evals_per_epoch))
        else:
            training_args.evaluation_strategy = "no"

    # Trainer
    if model_args.model_type == "huggingface":
        trainer = Seq2SeqTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None
        )

    elif model_args.model_type == "distilbart":
        trainer = Seq2SeqKDTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            use_kd_loss=model_args.use_kd_loss,
            teacher_model=teacher_model,
            temperature=2.0,
            alpha_data=model_args.alpha_data,
            alpha_logits=model_args.alpha_logits,
            alpha_hidden=model_args.alpha_hidden
        )

    else:
        trainer = Seq2SeqInterpolationTrainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            num_interpolation_epochs=model_args.num_interpolation_epochs,
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
        logging.info("*** Training ***")
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

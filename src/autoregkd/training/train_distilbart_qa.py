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
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
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

    max_length: Optional[int] = field(
        default=512,
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
        default=64,
        metadata={"help": "Maximum answer length allowed"}
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

    # Update output dir if necessary
    if data_args.use_v2:
        training_args.output_dir += "v2/"
    else:
        training_args.output_dir += "v1/"

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
    tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

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
    max_length = data_args.max_length
    padding = "max_length" if data_args.pad_to_max_length else False
    pad_on_right = tokenizer.padding_side == "right"

    def preprocess_squad_train(examples):
        """
        Pre-process SQuAD examples for training
        """
        questions = examples["question"]
        contexts = examples["context"]

        # Tokenize questions and contexts together
        tokenized_examples = tokenizer(
            questions if pad_on_right else contexts,
            contexts if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",    # Only truncate contexts
            padding=padding,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            stride=data_args.doc_stride
        )

        # Map features to their corresponding examples (when splitted)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # Map tokens to their positions in the original contexts
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Add the answer spans (start + end positions)
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # Get index of CLS token
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Sequence ids (context vs. question)
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Current example and answers
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # No answer case
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Make sure we start at the beginning of the context
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # Make sure we end at the end of the current span of the context
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Check for out of span
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Move the tokens until they match the start and end chars
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def preprocess_squad_eval(examples):
        """
        Preprocess validation examples. Similar to preprocess_squad_train but add features to retrieve spans of text

        """
        questions = examples["question"]
        contexts = examples["context"]

        # Tokenize questions and contexts together
        tokenized_examples = tokenizer(
            questions if pad_on_right else contexts,
            contexts if pad_on_right else questions,
            truncation="only_second" if pad_on_right else "only_first",  # Only truncate contexts
            padding=padding,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            stride=data_args.doc_stride
        )

        # Map features to their corresponding examples (when splitted)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # Examples' ids
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Sequence ids (context vs. question)
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # Current example
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

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

    if training_args.do_eval:
        if not eval_dataset:
            raise ValueError("No eval dataset available.")
            return

        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

        eval_dataset = eval_dataset.map(
            preprocess_squad_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache
        )

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

    # Data collator
    data_collator = default_data_collator

    # Eval steps (should be ~4 times per epoch)
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
        data_collator=data_collator
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

    # Load metric to use in evaluation
    if data_args.task == "question-answering":
        metric_name = "squad_v2" if data_args.use_v2 else "squad"  # EM/F1 scores
    else:
        raise ValueError("Unsupported task.")

    metric = load_metric(metric_name)

    # Hyper-parameters to use during the post-preprocessing step
    n_best_size = data_args.n_best_size
    max_answer_length = data_args.max_answer_length

    def postprocess_squad(examples, features, raw_predictions):
        # Logging.
        logging.info("Post-processing {} example predictions split into {} features."
                     .format(len(examples), len(features)))

        # Unpack start/end logits from the predictions
        all_start_logits, all_end_logits, _ = raw_predictions

        # Map example id to feature index
        example_id_to_index = {
            k: i for i, k in enumerate(examples["id"])
        }
        features_per_example = defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        predictions = OrderedDict()

        for example_index, example in enumerate(tqdm(examples)):
            feature_indices = features_per_example[example_index]

            min_null_score = None
            valid_answers = []

            # Context of the current example
            context = example["context"]

            for feature_index in feature_indices:
                # Logits for the current feature
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                # Current offset mapping to map logits to spans of text in the context
                offset_mapping = features[feature_index]["offset_mapping"]

                # Minimum null prediction
                cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Sort and keep the `n_best_size` best start/end logits
                start_indices = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
                end_indices = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

                # Iterate through all possible combinations of start-end indices
                for start_index in start_indices:
                    for end_index in end_indices:
                        # Skip if either index is out of the offset mapping's range
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue

                        # Skip invalid combinations
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        # Get the position of the start and end characters
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append({
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char]
                        })

            # Have at least one valid answer
            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                best_answer = {
                    "score": 0.,
                    "text": ""
                }

            # If use SQuADv2, we need to consider if the example is impossible to answer
            if data_args.use_v2:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

            # SQuADv1 case
            else:
                predictions[example["id"]] = best_answer["text"]

        # Format predictions to compute metrics
        if data_args.use_v2:
            formatted_predictions = [{
                "id": k,
                "prediction_text": v,
                "no_answer_probability": 0.0
            } for k, v in predictions.items()]
        else:
            formatted_predictions = [{
                "id": k,
                "prediction_text": v
            } for k, v in predictions.items()]

        # References
        references = [{
            "id": ex["id"],
            "answers": ex["answers"]
        } for ex in examples]

        return metric.compute(predictions=formatted_predictions, references=references)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logging.info("*** Evaluating ***")

        # Make predictions
        raw_predictions = trainer.predict(eval_dataset, metric_key_prefix="eval")

        # Recover the columns hidden by the trainer
        eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        # Postprocess to obtain final metrics
        metrics = postprocess_squad(examples=datasets["validation"],
                                    features=eval_dataset,
                                    raw_predictions=raw_predictions.predictions)

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
        print(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:
        logging.info("*** Testing ***")

        # Raw predictions
        raw_predictions = trainer.predict(test_dataset, metric_key_prefix="test")

        # Recover the columns hidden by the trainer
        test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        # Postprocess to obtain final metrics
        metrics = postprocess_squad(examples=datasets["test"],
                                    features=test_dataset,
                                    raw_predictions=raw_predictions.predictions)

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

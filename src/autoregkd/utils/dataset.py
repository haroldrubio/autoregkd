from os import read
import sys
import torch
import json
import nltk
from filelock import FileLock
import numpy as np
from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    default_data_collator,
    DataCollatorWithPadding
)
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from datasets import load_dataset, load_metric
with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)

from .dataset_utils import postprocess_qa_predictions

class ConvAIDataset(Dataset):
    def __init__(self, root_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class QA_Dataset():
    def __init__(self, training_args, model_args, data_args) -> None:
        # BART tokenizer
        self.tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name)
        self.pad_on_right = self.tokenizer.padding_side == "right"

        # Store args
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        # Load dataset
        datasets = load_dataset(data_args.dataset_name)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['validation'] if 'validation' in datasets.keys() else None
        self.raw_val_dataset = datasets['validation']  if 'validation' in datasets.keys() else None

        # Get column names
        self.train_column_names = self.train_dataset.column_names
        self.val_column_names = self.val_dataset.column_names
    
        self.metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")
    
    def access_datasets(self):
        train_dataset = None
        if self.train_dataset:
            if self.data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.data_args.max_train_samples))
            column_names = self.train_column_names
            train_dataset = self.train_dataset.map(
                self._pre_process_train,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache
            )

        val_dataset = None
        if self.val_dataset:
            if self.data_args.max_val_samples is not None:
                self.val_dataset = self.val_dataset.select(range(self.data_args.max_val_samples))
            val_dataset = self.val_dataset.map(
                self._pre_process_valid,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
            )

        # Data collator
        data_collator = (
            default_data_collator
            if self.data_args.pad_to_max_length
            else DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None)
        )
        
        return train_dataset, val_dataset, data_collator
    
    def postprocess_text(self, examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.training_args.output_dir,
        )
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.val_dataset]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_metrics(self, p: EvalPrediction):
        return self.metric.compute(predictions=p.predictions, references=p.label_ids)
    
    def _pre_process_valid(self, examples):
        column_names = self.val_column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if self.pad_on_right else context_column_name],
            examples[context_column_name if self.pad_on_right else question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
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
            context_index = 1 if self.pad_on_right else 0

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

    def _pre_process_train(self, examples):
        column_names = self.train_column_names
        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples[question_column_name if self.pad_on_right else context_column_name],
            examples[context_column_name if self.pad_on_right else question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.data_args.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
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
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

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
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
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


class Gen_Dataset():
    def __init__(self, model_args, data_args) -> None:
        # BART tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)

        # Max lengths
        self.max_source_length = data_args.max_source_length
        self.max_target_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else False

        # Store data args
        self.data_args = data_args

        # Load datasets
        # Load dataset
        datasets = load_dataset(data_args.dataset_name)
        self.train_dataset = datasets['train']
        self.val_dataset = datasets['validation'] if 'validation' in datasets.keys() else None
        self.test_dataset = datasets['test'] if 'test' in datasets.keys() else None
        # Do train-test split for ConvAI2 since there's no validation split

        # Get column names
        self.column_names = self.train_dataset.column_names
        if self.val_dataset:
            assert self.column_names == self.val_dataset.column_names
        if self.test_dataset:
            assert self.column_names == self.test_dataset.column_names

        self.metric_name = None
        if data_args.task == "summarization":
            self.metric_name = "rouge"
        elif data_args.task == "dialogue-generation":
            self.metric_name = "f1"
        else:
            raise ValueError("Unsupported task")

        self.metric = load_metric(self.metric_name)
    
    def access_datasets(self):
        train_dataset = None
        if self.train_dataset:
            if self.data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(self.data_args.max_train_samples))
            train_dataset = self.train_dataset.map(
                self._preprocess_xsum,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache
            )

        val_dataset = None
        if self.val_dataset:
            if self.data_args.max_val_samples is not None:
                self.val_dataset = self.val_dataset.select(range(self.data_args.max_val_samples))
            val_dataset = self.val_dataset.map(
                self._preprocess_xsum,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache
            )

        test_dataset = None
        if self.test_dataset:
            if self.data_args.max_test_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.data_args.max_test_samples))
            test_dataset = self.test_dataset.map(
                self._preprocess_xsum,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache
            )
        
        # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if self.data_args.pad_to_max_length:
            data_collator = self.default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=None,
            )
        
        return train_dataset, val_dataset, test_dataset, data_collator

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Apply softmax to predictions
        preds = torch.tensor(preds)
        preds = F.softmax(preds, dim=2)
        preds = preds.argmax(dim=2)

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels
        if self.data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Preprocess
        decoded_preds, decoded_labels = self._postprocess_text(decoded_preds, decoded_labels)

        if self.metric_name == "rouge":
            result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = {}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {key: round(value, 4) for key, value in result.items()}

        return result

    def _postprocess_text(self, preds, labels):
        if self.data_args.task == "summarization":
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def _preprocess_xsum(self, examples):
        inputs = examples["document"]
        targets = examples["summary"]

        # Tokenize source
        model_inputs = self.tokenizer(inputs, max_length=self.max_source_length, padding=self.padding, truncation=True)

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


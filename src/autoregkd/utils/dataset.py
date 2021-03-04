from os import read
import sys
import torch
import json
import nltk
from filelock import FileLock
import numpy as np
from transformers import (
    BartTokenizer,
    DataCollatorForSeq2Seq
)
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from datasets import load_dataset, load_metric
with FileLock(".lock") as lock:
    nltk.download("punkt", quiet=True)


from .dataset_utils import load_squad
# Data handling goes here
class SquadDataset(Dataset):
    def __init__(self, root_dir, debug=False):
        self.encodings = load_squad(root_dir, debug)

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

class ConvAIDataset(Dataset):
    def __init__(self, root_dir):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

class HF_Dataset():
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
        elif data_args.task == "question-answering":
            self.metric_name = "f1"
        elif data_args.task == "ialogue-generation":
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

    def postprocess_text(self, preds, labels):
        if self.data_args.task == "summarization":
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

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
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        if self.metric_name == "rouge":
            result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = {}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {key: round(value, 4) for key, value in result.items()}

        return result

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


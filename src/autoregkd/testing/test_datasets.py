import sys
import unittest
import torch
import numpy as np

from transformers import (
    BartModel,
    TrainingArguments
)
from src.autoregkd.testing.test_utils import (
    hf_data,
    ModelArguments
)
from src.autoregkd.utils.custom_args import DatasetArguments
from src.autoregkd.utils.dataset import QA_Dataset
from src.autoregkd.models.custom_bart import(
    InterpolationModule,
    DistilBartConfig,
    DistilBartDecoder,
    InterpolationDecoder,
    DistilModelOutputWithPastAndCrossAttentions
)


class TestDatasets(unittest.TestCase):
    def setUp(self):
        training_args = TrainingArguments(output_dir='test_ha')
        training_args.do_eval = True
        training_args.do_train = True
        model_args = ModelArguments()
        data_args = DatasetArguments(task='question-answering', dataset_name='squad_v2')
        self.hf_train, self.hf_val, self.hf_collator = hf_data()
        self.ha_train, self.ha_val, self.ha_collator = QA_Dataset(training_args, model_args, data_args).access_datasets()
    
    def test_collator(self):
        self.assertEqual(self.ha_collator, self.hf_collator)

    def test_train(self):
        self.assertEqual(len(self.hf_train), len(self.ha_train))
        for hf, ha in zip(self.hf_train, self.ha_train):
            self.assertEqual(hf.keys(), ha.keys())
            keys = hf.keys()
            for k in keys:
                self.assertEqual(hf[k], ha[k])
            

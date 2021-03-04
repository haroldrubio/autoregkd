# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model. """
import copy
from typing import Optional
import torch.nn.functional as F
from torch import nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartModel
)
from transformers.utils import logging
from transformers.utils.dummy_pt_objects import BartForConditionalGeneration, BartPretrainedModel

class DistilBartConfig(BartConfig):
    def __init__(self,
                 encoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 decoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 model_name = 'facebook/bart-large',
                 vocab_size=50265,
                 max_position_embeddings=1024,
                 encoder_ffn_dim=4096,
                 encoder_attention_heads=16,
                 decoder_ffn_dim=4096,
                 decoder_attention_heads=16,
                 encoder_layerdrop=0.0,
                 decoder_layerdrop=0.0,
                 activation_function="gelu",
                 d_model=1024,
                 dropout=0.1,
                 attention_dropout=0.0,
                 activation_dropout=0.0,
                 init_std=0.02,
                 classifier_dropout=0.0,
                 scale_embedding=False,
                 gradient_checkpointing=False,
                 use_cache=True,
                 num_labels=3,
                 pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 is_encoder_decoder=True,
                 decoder_start_token_id=2,
                 forced_eos_token_id=2,
                 **kwargs
                 ):
        super().__init__(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            encoder_ffn_dim=encoder_ffn_dim,
            encoder_attention_heads=encoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            activation_function=activation_function,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            init_std=init_std,
            classifier_dropout=classifier_dropout,
            scale_embedding=scale_embedding,
            gradient_checkpointing=gradient_checkpointing,
            use_cache=use_cache,
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.model_name = model_name
        if model_name == 'facebook/bart-large':
            self.encoder_layer_indices = encoder_layer_indices
            self.decoder_layer_indices = decoder_layer_indices
            self.encoder_layers = len(self.encoder_layer_indices)
            self.decoder_layers = len(self.decoder_layer_indices)
        elif model_name == 'facebook/bart-base':
            self.encoder_layer_indices = [0, 1, 2, 3, 4, 5]
            self.decoder_layer_indices = [0, 1, 2, 3, 4, 5]
            self.encoder_layers = len(self.encoder_layer_indices)
            self.decoder_layers = len(self.decoder_layer_indices)
    
    def set_distillation(self, encoder_layer_indices, decoder_layer_indices):
        self.encoder_layer_indices = encoder_layer_indices
        self.decoder_layer_indices = decoder_layer_indices
        self.encoder_layers = len(self.encoder_layer_indices)
        self.decoder_layers = len(self.decoder_layer_indices)

class DistilBartEncoder(BartEncoder):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_encoder: BartEncoder,
                 embed_tokens: Optional[nn.Embedding] = None
                 ):
        super().__init__(config=config, embed_tokens=embed_tokens)
        self.layers = nn.ModuleList()
        for i in config.encoder_layer_indices:
            self.layers.append(copy.deepcopy(bart_encoder.layers[i]))


class DistilBartDecoder(BartDecoder):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_decoder: BartDecoder,
                 embed_tokens: Optional[nn.Embedding] = None
                 ):
        super().__init__(config=config, embed_tokens=embed_tokens)
        self.layers = nn.ModuleList()
        for i in config.decoder_layer_indices:
            self.layers.append(copy.deepcopy(bart_decoder.layers[i]))


class DistilBart(BartModel):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_model: BartModel
                 ):
        super().__init__(config)

        self.encoder = DistilBartEncoder(config=config, bart_encoder=bart_model.encoder, embed_tokens=self.shared)
        self.decoder = DistilBartDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)

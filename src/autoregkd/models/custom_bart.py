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
from dataclasses import dataclass
import torch
from typing import Optional
import torch.nn.functional as F
from torch import nn
import random
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models import bart
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartModel
)

class DistilBartConfig(BartConfig):
    def __init__(self,
                 encoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 decoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 swap_prob=0,
                 model_name = 'facebook/bart-large',
                 **kwargs
                 ):
        super().__init__(
            **kwargs
        )
        self.swap_prob = swap_prob
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
        # Copy structural layers and some of the transformer layers
        self.layers = nn.ModuleList()
        for i in config.encoder_layer_indices:
            self.layers.append(copy.deepcopy(bart_encoder.layers[i]))
        self.embed_tokens = copy.deepcopy(bart_encoder.embed_tokens)
        self.embed_positions = copy.deepcopy(bart_encoder.embed_positions)
        self.layernorm_embedding = copy.deepcopy(bart_encoder.layernorm_embedding)

class DistilBartDecoder(BartDecoder):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_decoder: BartDecoder,
                 embed_tokens: Optional[nn.Embedding] = None
                 ):
        super().__init__(config=config, embed_tokens=embed_tokens)
        # Copy structural layers and some of the transformer layers
        self.layers = nn.ModuleList()
        for i in config.decoder_layer_indices:
            self.layers.append(copy.deepcopy(bart_decoder.layers[i]))
        self.embed_tokens = copy.deepcopy(bart_decoder.embed_tokens)
        self.embed_positions = copy.deepcopy(bart_decoder.embed_positions)
        self.layernorm_embedding = copy.deepcopy(bart_decoder.layernorm_embedding)


class DistilBart(BartModel):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_model: BartModel,
                 decoder_type: str = None,
                 ):
        super().__init__(config)
        self.shared = bart_model.shared
        self.encoder = DistilBartEncoder(config=config, bart_encoder=bart_model.encoder, embed_tokens=self.shared)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.shared.parameters():
            param.requires_grad = False

        # Set decoder
        if decoder_type == None:
            self.decoder = DistilBartDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)
        elif decoder_type == 'interpolation':
            self.decoder = InterpolationDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)

class InterpolationModule(nn.Module):
    """
    This module contains no parameters and performs a swapping operation on the hidden unit level
    between two inputs of the same shape
    """
    def __init__(self, swap_prob=0.5):
        super().__init__()
        self.swap_prob = swap_prob
        self.register_buffer("swap_probability", self.swap_prob)
    def forward(self, parent_in, student_in):
        """
            Args:
                parent_in (torch.tensor): An input tensor from path 1
                student_in (torch.tensor): An input tensor from path 2
            Returns:
                (parent_out, student_out) (tuple): The interpolated hidden states
        """
        swap_prob = self.swap_prob
        # Obtain a common shape
        common_shape = parent_in.shape
        assert common_shape == student_in.shape

        # Generate mask
        rand_tensor = torch.rand(common_shape)
        swapping_mask = torch.zeros(common_shape)
        swapping_mask[rand_tensor <= swap_prob] = 1
        staying_mask = torch.abs(swapping_mask - 1)
        del rand_tensor

        # Create two output tensors
        parent_out = staying_mask * parent_in + swapping_mask * student_in
        student_out = staying_mask * student_in + swapping_mask * parent_in

        return (parent_out, student_out)

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

class InterpolationDecoder(DistilBartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: DistilBartConfig, bart_decoder: BartDecoder, embed_tokens: Optional[nn.Embedding] = None):
        # Copy structural layers and some of the transformer layers into the student
        self.decoder_layer_indices = config.decoder_layer_indices
        self.std_layers = nn.ModuleList()
        for i in config.decoder_layer_indices:
            self.std_layers.append(copy.deepcopy(bart_decoder.layers[i]))
        self.std_embed_tokens = copy.deepcopy(bart_decoder.embed_tokens)
        self.std_embed_positions = copy.deepcopy(bart_decoder.embed_positions)
        self.std_layernorm_embedding = copy.deepcopy(bart_decoder.layernorm_embedding)

        # Copy structural layers and ALL transformer layers into the teacher
        self.layers = nn.ModuleList()
        for layer in bart_decoder.layers:
            self.layers.append(copy.deepcopy(layer))
        self.embed_tokens = copy.deepcopy(bart_decoder.embed_tokens)
        self.embed_positions = copy.deepcopy(bart_decoder.embed_positions)
        self.layernorm_embedding = copy.deepcopy(bart_decoder.layernorm_embedding)

        # Freeze teacher parameters
        for l in self.layers:
            for p in l.parameters():
                p.requires_grad = False
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.embed_positions.parameters():
            p.requires_grad = False
        for p in self.layernorm_embedding.parameters():
            p.requires_grad = False

        # Decoder has one interpolation module per student layer minus 1
        # Since the final outputs are not interpolated
        self.interp = nn.ModuleList([InterpolationModule(config.swap_prob) for _ in range(config.decoder_layers - 1)])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # Harold: Branch off here
        std_input_embeds = inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            std_input_embeds = self.std_embed_tokens(input_ids) * self.embed_scale
          
        # Harold: No change in attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        # Harold: No change in cross attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        # Harold: Branch off positions
        positions = self.embed_positions(input_shape, past_key_values_length)
        std_positions = self.std_embed_positions(input_shape, past_key_values_length)

        # Harold: Create std hidden states
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        std_hidden_states = std_input_embeds + std_positions
        std_hidden_states = self.std_layernorm_embedding(std_hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        std_hidden_states = F.dropout(std_hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        # Harold: Accumulation of decoder states remains same
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        # Harold: decoder_idx counter
        # TODO: If not training, skip teacher passes entirely and iterate over student layers
        if not self.training:
            for idx, std_layer in enumerate(self.std_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue
                
                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if getattr(self.config, "gradient_checkpointing", False) and self.training:
                    if use_cache:
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, use_cache)

                        return custom_forward
                    # Harold: if arrived at layer aligned pair - perform a student pass
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(std_layer),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        encoder_head_mask[idx] if encoder_head_mask is not None else None,
                        None,
                    )
                else:
                    layer_outputs = std_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                # If not training, only std_hidden_states exist
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)
        else:
            std_parallel = self.decoder_layer_indices[0]
            interp_idx = 0
            for idx, decoder_layer in enumerate(self.layers):
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    if use_cache:
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, use_cache)

                        return custom_forward
                    # Harold: if arrived at layer aligned pair - perform a student pass
                    std_layer_outputs = None
                    if idx == std_parallel:
                        # Fetch student decoder layer
                        std_decoder_layer = self.std_layers[interp_idx]
                        std_layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(std_decoder_layer),
                        std_hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        encoder_head_mask[idx] if encoder_head_mask is not None else None,
                        None,
                    )
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states,
                        encoder_attention_mask,
                        head_mask[idx] if head_mask is not None else None,
                        encoder_head_mask[idx] if encoder_head_mask is not None else None,
                        None,
                    )
                else:
                    # Harold: Same as above for non gradient checkpoint case
                    std_layer_outputs = None
                    if idx == std_parallel:
                        # Fetch student decoder layer
                        std_decoder_layer = self.std_layers[interp_idx]
                        std_layer_outputs = std_decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                # TODO: If not training, only std_hidden_states exist
                if idx == std_parallel:
                    std_hidden_states = std_layer_outputs[0]
                hidden_states = layer_outputs[0]

                # Harold: insert interpolation after the forward passes
                # Check for layer alignment
                if idx == std_parallel:
                    # Check if interpolation module exists at this pairing
                    if interp_idx < len(self.interp):
                        # If it does, fetch the interpolation module
                        interp_module = self.interp[interp_idx]
                        hidden_states, std_hidden_states = interp_module(hidden_states, std_hidden_states)
                    # Step the indices
                    interp_idx += 1
                    if interp_idx < len(self.decoder_layer_indices):
                        std_parallel = self.decoder_layer_indices[interp_idx]

                if use_cache:
                    next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        # Harold: check for training, if not training then set std hidden states
        if not self.training:
            std_hidden_states = hidden_states
        # Harold: handle the parsing of last hidden states downstream by cutting the states in half
        return DistilModelOutputWithPastAndCrossAttentions(
            last_hidden_state=std_hidden_states,
            teacher_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

@dataclass
class DistilModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    teacher_hidden_state: torch.FloatTensor = None
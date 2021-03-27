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
import numpy as np
from re import L
import sys
from dataclasses import dataclass, fields
import torch
from typing import List, Optional, Iterable, Tuple
from torch._C import device
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import CrossEntropyLoss
import random
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartForQuestionAnswering, BartLearnedPositionalEmbedding,
    BartEncoder,
    BartDecoder,
    BartModel,
    Seq2SeqQuestionAnsweringModelOutput
)

class DistilBartConfig(BartConfig):
    def __init__(self,
                 swap_prob=0,
                 encoder_type: str = 'distill',
                 decoder_type: str = 'distill',
                 encoder_layer_indices: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 decoder_layer_indices: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 num_teacher_enc: int = 12,
                 num_teacher_dec: int = 12,
                 loss_type: str = 'finetune',
                 **kwargs
                 ):
        super().__init__(
            **kwargs
        )
        self.swap_prob = swap_prob
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_layer_indices = encoder_layer_indices
        self.decoder_layer_indices = decoder_layer_indices
        self.num_teacher_enc = num_teacher_enc
        self.num_teacher_dec = num_teacher_dec
        self.loss_type = loss_type

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
        # Freeze embeddings
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.embed_positions.parameters():
            param.requires_grad = False
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
        # Freeze embeddings
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.embed_positions.parameters():
            param.requires_grad = False
        self.layernorm_embedding = copy.deepcopy(bart_decoder.layernorm_embedding)


class DistilBart(BartModel):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_model: BartModel,
                 ):
        super().__init__(config)
        self.shared = bart_model.shared
        self.encoder = DistilBartEncoder(config=config, bart_encoder=bart_model.encoder, embed_tokens=self.shared)
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.shared.parameters():
            param.requires_grad = False

        # Set decoder
        if config.decoder_type == 'distilbart':
            self.decoder = DistilBartDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)
        elif config.decoder_type == 'interpolation':
            self.decoder = InterpolationDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)

class DistilBartForQuestionAnswering(BartForQuestionAnswering):
    def __init__(self, config: DistilBartConfig):
        super().__init__(config)
        # Handle decoder type
        if config.decoder_type == 'interpolate':
            self.model.decoder = InterpolationDecoder(config, self.model.shared)

        # Handle loss type
        self.loss_type = config.loss_type
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Harold: overwrite return dict - this lets us easily modify what gets returned
        return_dict = False

        if start_positions is not None and end_positions is not None:
            use_cache = False
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Harold: delete all but the first 2 items in outputs
        # del outputs[2:]

        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        std_loss = None
        teach_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            std_loss = (start_loss + end_loss) / 2

        # Harold: Handle interpolation loss - perform loss computation twice
        if self.loss_type == 'interpolate' and self.training:
            # Obtain teacher hidden states

            tch_sequence_output = outputs[1]    
            tch_logits = self.qa_outputs(tch_sequence_output)
            tch_start_logits, tch_end_logits = tch_logits.split(1, dim=-1)
            tch_start_logits = tch_start_logits.squeeze(-1)
            tch_end_logits = tch_end_logits.squeeze(-1)

            if start_positions is not None and end_positions is not None:
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                tch_start_loss = loss_fct(tch_start_logits, start_positions)
                tch_end_loss = loss_fct(tch_end_logits, end_positions)
                teach_loss = (tch_start_loss + tch_end_loss) / 2
                total_loss = (teach_loss + std_loss) / 2
        else:
            total_loss = std_loss

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

@dataclass
class DistilModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    teacher_hidden_state: torch.FloatTensor = None
# -----------------------------------------------
# -----------------------------------------------
# ---------Initial Interpolation Decoder---------
# -----------------------------------------------
# -----------------------------------------------
class InterpolationScheduler():
    def __init__(
        self,
        modules: List[nn.Module],
        sch_params: dict,
        num_training_steps: int
    ):
        '''
        Expect the dict to have the following format: keys "max_prob", "cool_down"
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.modules = modules
        self.num_training_steps = num_training_steps
        self.max_prob = sch_params['max_prob']
        self.cool_down = sch_params['cool_down']
        self.curr_step = 0
        # TODO: Keep running Python float list of slopes and constantly allocate a new tensor
        self.probs = list(np.zeros(len(modules)))

        # Decide cool down midpoints
        self.midpoints = [float((i + 1)/(len(modules) + 1)) for i in range(len(modules))]
        # Check validity of arguments
        max_prob, cool_down = sch_params['max_prob'], sch_params['cool_down']
        max_interval = 2 / (len(modules) + 1)
        assert cool_down < max_interval, "cool_down out of training bounds"


    def step(self):
        """
        Performs a single optimization step.
        """
        for idx, module in enumerate(self.modules):
            # Convert percentages of training to step number
            curr_midpoint = self.midpoints[idx]
            start_cd = int(self.num_training_steps * (curr_midpoint - (self.cool_down / 2)))
            end_cd = int(self.num_training_steps * (curr_midpoint + (self.cool_down / 2)))
            for p in module.parameters():
                # Determine where in the schedule this is
                if self.curr_step == start_cd:
                    # Starting cooldown
                    self.probs[idx] = self.max_prob
                elif self.curr_step > start_cd and self.curr_step < end_cd:
                    # In cooldown phase
                    slope = self.max_prob / (end_cd - start_cd)
                    self.probs[idx] -= slope
                elif self.curr_step == end_cd:
                    # Stop swapping
                    self.probs[idx] = 0
                # Write out next probability to GPU tensor
                p.data = torch.tensor(self.probs[idx], dtype=self.dtype, device=self.device)
                # Enforce: non-negativity
                if p.data < 0:
                    p.data *= -1

        self.curr_step += 1

class InterpolationModule(nn.Module):
    """
    This module contains no parameters and performs a swapping operation on the hidden unit level
    between two inputs of the same shape
    """
    def __init__(self, swap_prob=0):
        super().__init__()
        swap_prob = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.swap_prob = nn.Parameter(torch.tensor(swap_prob, device=self.device, dtype=torch.double))
        self.swap_prob.requires_grad = False
        self.register_parameter("swap_prob", self.swap_prob)
    def forward(self, parent_in, student_in):
        """
            Args:
                parent_in (torch.tensor): An input tensor from path 1
                student_in (torch.tensor): An input tensor from path 2
            Returns:
                (parent_out, student_out) (tuple): The interpolated hidden states
        """
        if self.training:
            swap_prob = self.swap_prob
        else:
            swap_prob = 0
        # Obtain a common shape
        common_shape = parent_in.shape
        assert common_shape == student_in.shape

        # Generate mask
        rand_tensor = torch.rand(common_shape, device=self.device)
        swapping_mask = torch.zeros(common_shape, device=self.device)
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

class InterpolationDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: DistilBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config=config, embed_tokens=embed_tokens)
        # Copy structural layers and some of the transformer layers into the student
        self.decoder_layer_indices = config.decoder_layer_indices
        # Initialize student layers to some subset of the teacher, these will be set later
        self.std_layers = nn.ModuleList([self.layers[i] for i in range(len(config.decoder_layer_indices))])
        if embed_tokens is not None:
            self.std_embed_tokens = embed_tokens
        else:
            self.std_embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.std_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.std_layernorm_embedding = nn.LayerNorm(config.d_model)
        # Decoder has one interpolation module per student layer minus 1
        # Since the final outputs are not interpolated
        self.interp = nn.ModuleList([InterpolationModule() for _ in range(len(config.decoder_layer_indices) - 1)])


    def setup_interpolation(self):
        """
        Wrapper function that should be called after teacher parameters are loaded in
        Loads in the embeddings, freezes the teacher highway, and student embeddings
        """
        self.load_std_embeds()
        self.freeze_std_embeds()
        self.freeze_teacher_layers()
        self.unfreeze_std_layers()

    def load_std_embeds(self):
        """ After the model has been initialized and teacher information has been loaded, initialize the student
           embeddings from the teacher, and freeze these embeddings """
        self.std_embed_tokens.load_state_dict(self.embed_tokens.state_dict())
        self.std_embed_positions.load_state_dict(self.embed_positions.state_dict())
        self.std_layernorm_embedding.load_state_dict(self.layernorm_embedding.state_dict())
    
    def freeze_teacher_layers(self):
        """ Freeze the teacher highway gradients """
        for l in self.layers:
            for p in l.parameters():
                p.requires_grad = False
    
    def freeze_std_embeds(self):
        """ Freeze the student copy of the embeddings """
        for p in self.std_embed_positions.parameters():
            p.requires_grad = False
        for p in self.std_embed_tokens.parameters():
            p.requires_grad = False
    
    def unfreeze_std_layers(self):
        for l in self.std_layers:
            for p in l.parameters():
                p.requires_grad = True

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
          
        # Harold: Maybe change in attention mask?
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        # Harold: Shared cross attention mask
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
        std_parallel = self.decoder_layer_indices[0]
        interp_idx = 0
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states and idx == std_parallel:
                all_hidden_states += (std_hidden_states,)
            
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
                if idx == std_parallel:
                    # Fetch student decoder layer
                    std_decoder_layer = self.std_layers[interp_idx]
                    std_layer_outputs = std_decoder_layer(
                    std_hidden_states,
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
                # TODO: Debug - skip interpolation
                
                # Check if interpolation module exists at this pairing
                
                if interp_idx < len(self.interp):
                    # If it does, fetch the interpolation module
                    interp_module = self.interp[interp_idx]
                    hidden_states, std_hidden_states = interp_module(hidden_states, std_hidden_states)
                
                # Step the indices
                interp_idx += 1
                if interp_idx < len(self.decoder_layer_indices):
                    std_parallel = self.decoder_layer_indices[interp_idx]
                
                # Maintain student history
                if use_cache:
                    next_decoder_cache += (std_layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (std_layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (std_layer_outputs[2],)

        # Harold: only add if last layers are aligned
        # add hidden states from the last decoder layer
        if output_hidden_states and idx == std_parallel:
            all_hidden_states += (std_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [std_hidden_states, hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        # Harold: handle the parsing of last hidden states
        return DistilModelOutputWithPastAndCrossAttentions(
            last_hidden_state=std_hidden_states,
            teacher_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
# -----------------------------------------------
# -----------------------------------------------
# ---------Initial Interpolation Decoder---------
# -----------------------------------------------
# -----------------------------------------------

# -----------------------------------------------
# -----------------------------------------------
# -----------Interpolation Decoder V2s-----------
# -----------------------------------------------
# -----------------------------------------------
class InterpolationSchedulerV2s():
    def __init__(
        self,
        modules: List[nn.Module],
        sch_params: dict,
        num_training_steps: int
    ):
        '''
        Expect the dict to have the following format: keys "max_prob", "cool_down"
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.modules = modules
        self.num_training_steps = num_training_steps
        self.max_prob = sch_params['max_prob']
        self.cool_down = sch_params['cool_down']
        self.curr_step = 0
        # TODO: Keep running Python float list of slopes and constantly allocate a new tensor
        self.probs = list(np.zeros(len(modules)))

        # Decide cool down midpoints
        self.midpoints = [float((i + 1)/(len(modules) + 1)) for i in range(len(modules))]
        # Check validity of arguments
        max_prob, cool_down = sch_params['max_prob'], sch_params['cool_down']
        max_interval = 2 / (len(modules) + 1)
        assert cool_down < max_interval, "cool_down out of training bounds"


    def step(self):
        """
        Performs a single optimization step.
        """
        for idx, module in enumerate(self.modules):
            # Convert percentages of training to step number
            curr_midpoint = self.midpoints[idx]
            start_cd = int(self.num_training_steps * (curr_midpoint - (self.cool_down / 2)))
            end_cd = int(self.num_training_steps * (curr_midpoint + (self.cool_down / 2)))
            for p in module.parameters():
                # Determine where in the schedule this is
                if self.curr_step == start_cd:
                    # Starting cooldown
                    self.probs[idx] = self.max_prob
                elif self.curr_step > start_cd and self.curr_step < end_cd:
                    # In cooldown phase
                    slope = self.max_prob / (end_cd - start_cd)
                    self.probs[idx] -= slope
                elif self.curr_step == end_cd:
                    # Stop swapping
                    self.probs[idx] = 0
                # Write out next probability to GPU tensor
                p.data = torch.tensor(self.probs[idx], dtype=self.dtype, device=self.device)
                # Enforce: non-negativity
                if p.data < 0:
                    p.data *= -1

        self.curr_step += 1

class InterpolationModule(nn.Module):
    """
    This module contains no parameters and performs a swapping operation on the hidden unit level
    between two inputs of the same shape
    """
    def __init__(self, swap_prob=0):
        super().__init__()
        swap_prob = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.swap_prob = nn.Parameter(torch.tensor(swap_prob, device=self.device, dtype=torch.double))
        self.swap_prob.requires_grad = False
        self.register_parameter("swap_prob", self.swap_prob)
    def forward(self, parent_in, student_in):
        """
            Args:
                parent_in (torch.tensor): An input tensor from path 1
                student_in (torch.tensor): An input tensor from path 2
            Returns:
                (parent_out, student_out) (tuple): The interpolated hidden states
        """
        if self.training:
            swap_prob = self.swap_prob
        else:
            swap_prob = 0
        # Obtain a common shape
        common_shape = parent_in.shape
        assert common_shape == student_in.shape

        # Generate mask
        rand_tensor = torch.rand(common_shape, device=self.device)
        swapping_mask = torch.zeros(common_shape, device=self.device)
        swapping_mask[rand_tensor <= swap_prob] = 1
        staying_mask = torch.abs(swapping_mask - 1)
        del rand_tensor

        # Create two output tensors
        parent_out = staying_mask * parent_in + swapping_mask * student_in
        student_out = staying_mask * student_in + swapping_mask * parent_in

        return (parent_out, student_out)

class InterpolationDecoder(BartDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: DistilBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config=config, embed_tokens=embed_tokens)
        # Copy structural layers and some of the transformer layers into the student
        self.decoder_layer_indices = config.decoder_layer_indices
        # Initialize student layers to some subset of the teacher, these will be set later
        self.std_layers = nn.ModuleList([self.layers[i] for i in range(len(config.decoder_layer_indices))])
        if embed_tokens is not None:
            self.std_embed_tokens = embed_tokens
        else:
            self.std_embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.std_embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.std_layernorm_embedding = nn.LayerNorm(config.d_model)
        # Decoder has one interpolation module per student layer minus 1
        # Since the final outputs are not interpolated
        self.interp = nn.ModuleList([InterpolationModule() for _ in range(len(config.decoder_layer_indices) - 1)])


    def setup_interpolation(self):
        """
        Wrapper function that should be called after teacher parameters are loaded in
        Loads in the embeddings, freezes the teacher highway, and student embeddings
        """
        self.load_std_embeds()
        self.freeze_std_embeds()
        self.freeze_teacher_layers()
        self.unfreeze_std_layers()

    def load_std_embeds(self):
        """ After the model has been initialized and teacher information has been loaded, initialize the student
           embeddings from the teacher, and freeze these embeddings """
        self.std_embed_tokens.load_state_dict(self.embed_tokens.state_dict())
        self.std_embed_positions.load_state_dict(self.embed_positions.state_dict())
        self.std_layernorm_embedding.load_state_dict(self.layernorm_embedding.state_dict())
    
    def freeze_teacher_layers(self):
        """ Freeze the teacher highway gradients """
        for l in self.layers:
            for p in l.parameters():
                p.requires_grad = False
    
    def freeze_std_embeds(self):
        """ Freeze the student copy of the embeddings """
        for p in self.std_embed_positions.parameters():
            p.requires_grad = False
        for p in self.std_embed_tokens.parameters():
            p.requires_grad = False
    
    def unfreeze_std_layers(self):
        for l in self.std_layers:
            for p in l.parameters():
                p.requires_grad = True

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
          
        # Harold: Maybe change in attention mask?
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        # Harold: Shared cross attention mask
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
        std_parallel = self.decoder_layer_indices[0]
        interp_idx = 0
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states and idx == std_parallel:
                all_hidden_states += (std_hidden_states,)
            
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
                if idx == std_parallel:
                    # Fetch student decoder layer
                    std_decoder_layer = self.std_layers[interp_idx]
                    std_layer_outputs = std_decoder_layer(
                    std_hidden_states,
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
                # TODO: Debug - skip interpolation
                
                # Check if interpolation module exists at this pairing
                
                if interp_idx < len(self.interp):
                    # If it does, fetch the interpolation module
                    interp_module = self.interp[interp_idx]
                    hidden_states, std_hidden_states = interp_module(hidden_states, std_hidden_states)
                
                # Step the indices
                interp_idx += 1
                if interp_idx < len(self.decoder_layer_indices):
                    std_parallel = self.decoder_layer_indices[interp_idx]
                
                # Maintain student history
                if use_cache:
                    next_decoder_cache += (std_layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (std_layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (std_layer_outputs[2],)

        # Harold: only add if last layers are aligned
        # add hidden states from the last decoder layer
        if output_hidden_states and idx == std_parallel:
            all_hidden_states += (std_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [std_hidden_states, hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        # Harold: handle the parsing of last hidden states
        return DistilModelOutputWithPastAndCrossAttentions(
            last_hidden_state=std_hidden_states,
            teacher_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
# -----------------------------------------------
# -----------------------------------------------
# -----------Interpolation Decoder V2s-----------
# -----------------------------------------------
# -----------------------------------------------
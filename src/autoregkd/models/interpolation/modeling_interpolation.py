import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from transformers import (
    BartConfig,
    BartForQuestionAnswering,
    BartForConditionalGeneration
)
from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer,
    BartDecoderLayer,
    BartEncoder,
    BartDecoder,
    BartModel,
    _expand_mask,
    shift_tokens_right
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput
)
from transformers.utils import logging

from ..distilbart.configuration_distilbart import DistilBartConfig

logger = logging.get_logger(__name__)


class LinearInterpolationModule(nn.Module):
    """

    """

    def __init__(self,
                 p: float = 0.0,
                 learnable_p: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0.0 and 1.0 inclusive. Got {}".format(p))

        self.learnable_p = learnable_p
        if self.learnable_p:
            self.p = Parameter(torch.normal(mean=p, std=0.01, size=()))
            self.p.requires_grad = True
        else:
            self.p = Parameter(torch.tensor(p, dtype=torch.float32))
            self.p.requires_grad = False

    def forward(self, student_in, teacher_in):
        """
        """
        if self.training:
            assert student_in.shape == teacher_in.shape
            if self.learnable_p:
                self.p = Parameter(torch.clamp(self.p, min=0.0, max=1.0))

            return self.p * student_in + (1 - self.p) * teacher_in
        else:
            return student_in

class TheseusInterpolationModule(nn.Module):
    """
    This module contains no parameters and performs a swapping operation on the hidden unit level
    between two inputs of the same shape
    """
    def __init__(self,
                 p: float = 0.0,
                 learnable_p: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0.0 and 1.0 inclusive. Got {}".format(p))
        self.p = Parameter(torch.tensor(p, dtype=torch.float32))
        self.p.requires_grad = False

    def forward(self, student_in, teacher_in):
        """
        """
        if self.training:
            assert student_in.shape == teacher_in.shape
            coin = random.random()
            if coin <= self.p:
                return teacher_in
            else:
                return student_in
        else:
            return student_in
class StochasticInterpolationModule(nn.Module):
    """
    This module contains no parameters and performs a swapping operation on the hidden unit level
    between two inputs of the same shape
    """
    def __init__(self,
                 p: float = 0.0,
                 learnable_p: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0.0 and 1.0 inclusive. Got {}".format(p))

        self.p = Parameter(torch.tensor(p, dtype=torch.float32))
        self.p.requires_grad = False

    def forward(self, student_in, teacher_in):
        """
        """
        if self.training:
            assert student_in.shape == teacher_in.shape
            common_shape = student_in.shape
            # Generate mask
            rand_tensor = torch.rand(common_shape).to(student_in.device)
            swapping_mask = torch.zeros(common_shape).to(student_in.device)
            swapping_mask[rand_tensor <= self.p] = 1
            staying_mask = torch.abs(swapping_mask - 1)
            del rand_tensor

            # Create two output tensors
            return swapping_mask * student_in + staying_mask * teacher_in
        else:
            return student_in

class RandomStochasticInterpolationModule(nn.Module):
    """
    Performs the same as stochastic interpolation except it scrambles the dimensions 
    of the teacher
    """
    def __init__(self,
                 p: float = 0.0,
                 learnable_p: bool = False):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("p must be between 0.0 and 1.0 inclusive. Got {}".format(p))

        self.p = Parameter(torch.tensor(p, dtype=torch.float32))
        self.p.requires_grad = False

    def forward(self, student_in, teacher_in):
        """
        """
        if self.training:
            assert student_in.shape == teacher_in.shape
            # Scrambling dimensions
            _, _, hid_dim = teacher_in.shape
            idxs = torch.tensor(random.sample(range(hid_dim), k=hid_dim), dtype=torch.long, device=teacher_in.device)
            last_dim = len(teacher_in.shape) - 1

            teacher_in = torch.index_select(teacher_in, last_dim, idxs)
            common_shape = student_in.shape
            # Generate mask
            rand_tensor = torch.rand(common_shape).to(student_in.device)
            swapping_mask = torch.zeros(common_shape).to(student_in.device)
            swapping_mask[rand_tensor <= self.p] = 1
            staying_mask = torch.abs(swapping_mask - 1)
            del rand_tensor

            # Create two output tensors
            return swapping_mask * student_in + staying_mask * teacher_in
        else:
            return student_in


class InterpolationScheduler:
    """

    """

    def __init__(self,
                 interpolation_modules: nn.ModuleList,
                 num_interpolation_steps: int,
                 max_prob: float = 1.0,
                 per_level_annealing_duration: float = 1.0,
                 step_size: int = 1
                 ):
        if max_prob < 0.0 or max_prob > 1.0:
            raise ValueError("Invalid max probability for interpolation module - should be between 0.0 and 1.0")
        if per_level_annealing_duration < 0.0 or per_level_annealing_duration > 1.0:
            raise ValueError("Invalid per-level annealing duration - should be between 0.0 and 1.0")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = interpolation_modules

        # Number of steps for for each interpolation module
        self.max_prob = max_prob
        self.per_level_annealing_steps = max(math.ceil((num_interpolation_steps - 1) * per_level_annealing_duration), 1)
        self.step_size = step_size
        self.slopes = [(self.max_prob - self.modules[i].p.item()) / self.per_level_annealing_steps * self.step_size for i in range(len(self.modules))]

        # Compute starting point for each interpolation module
        self.starting_points = []
        for i in range(len(self.modules)):
            if i == 0:
                self.starting_points.append(0)
            else:
                curr_starting_point = i * (num_interpolation_steps - 1 - self.per_level_annealing_steps) / (len(self.modules) - 1)
                self.starting_points.append(math.floor(curr_starting_point))
        self.starting_points.reverse()

        # Keep track of current states
        self.state = {}

    def step(self):
        """
        Perform a single step to update the interpolation modules' probabiltiies
        """
        if len(self.state) == 0:
            self.state = {
                "step": 0,
                "probs": [module.p.item() for module in self.modules]
            }

        # Increment step
        self.state["step"] += 1

        for i, module in enumerate(self.modules):
            if self.state["step"] < self.starting_points[i]:
                self.state["probs"][i] = self.state["probs"][i]
            elif self.state["step"] <= self.starting_points[i] + self.per_level_annealing_steps:
                if self.state["step"] % self.step_size == 0:
                    self.state["probs"][i] = min(1., self.state["probs"][i] + self.slopes[i])
                else:
                    self.state["probs"][i] = self.state["probs"][i]
            else:
                self.state["probs"][i] = 1.

            for p in module.parameters():
                p.data = torch.tensor(self.state["probs"][i], dtype=torch.float32, device=self.device)


class InterpolationBartEncoder(BartEncoder):
    def load_weights_to_student(self):
        pass

    def freeze_weights(self, freeze_embeddings):
        pass


class InterpolationBartDecoder(BartDecoder):
    """

    """

    def __init__(self, config: DistilBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config=config, embed_tokens=embed_tokens)
        # Set up config variables
        self.student_layer_indices = config.student_decoder_layer_indices
        self.student_decoder_layers = config.student_decoder_layers
        assert len(self.student_layer_indices) == self.student_decoder_layers

        # Initialize student's embeddings
        self.student_embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.student_embed_positions = BartLearnedPositionalEmbedding(config.max_position_embeddings, config.d_model)

        # Create student's decoder layers
        self.student_layers = nn.ModuleList([BartDecoderLayer(config=config) for _ in range(self.student_decoder_layers)])

        # Create student's layer-norm embedding
        self.student_layernorm_embedding = nn.LayerNorm(config.d_model)

        # Interpolation modules
        if config.interpolation_type == "stochastic":
            self.interpolation_modules = nn.ModuleList([
                StochasticInterpolationModule(p=config.interpolation_p, learnable_p=config.learnable_p)
                for _ in range(self.student_decoder_layers)
            ])
        elif config.interpolation_type == "random-stochastic":
            self.interpolation_modules = nn.ModuleList([
                RandomStochasticInterpolationModule(p=config.interpolation_p, learnable_p=config.learnable_p)
                for _ in range(self.student_decoder_layers)
            ])
        elif config.interpolation_type == "theseus":
            self.interpolation_modules = nn.ModuleList([
                TheseusInterpolationModule(p=config.interpolation_p, learnable_p=config.learnable_p)
                for _ in range(self.student_decoder_layers)
            ])
        assert len(self.student_layers) == len(self.interpolation_modules)

    def load_weights_to_student(self):
        # Copy embeddings
        self.student_embed_tokens.load_state_dict(self.embed_tokens.state_dict())
        self.student_embed_positions.load_state_dict(self.embed_positions.state_dict())

        # Copy decoder layers
        for student_idx, teacher_idx in enumerate(self.student_layer_indices):
            self.student_layers[student_idx].load_state_dict(self.layers[teacher_idx].state_dict())

        # Copy layernorm embedding
        self.student_layernorm_embedding.load_state_dict(self.layernorm_embedding.state_dict())

    def freeze_weights(self, freeze_embedding):
        # Freeze the entire teacher model first
        for layer in [self.embed_tokens, self.embed_positions, self.layers, self.layernorm_embedding]:
            for p in layer.parameters():
                p.requires_grad = False

        if freeze_embedding:
            for layer in [self.student_embed_tokens, self.student_embed_positions, self.student_layernorm_embedding]:
                for p in layer.parameters():
                    p.requires_grad = False

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
        r"""
        The inputs and behaviour are identical to the original model. However, computations are added for the student
        (which are identical to those of the teacher).
        For details of the arguments, please refer to the Huggingface's documentations.
        """
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

        # Branch-off starts here. For most of the computations, we repeat the same process for student model unless
        # stated otherwise in the comment
        # Variables with "student_" prefix are for the student model and those without prefix are for the teacher model
        # The student and teacher models should have the same inputs since the embeddings are frozen
        # However, we still start branching-off here in case we want to unfreeze the embeddings in the future
        student_attention_mask = attention_mask
        if self.training:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if inputs_embeds is None:
            student_inputs_embeds = self.student_embed_tokens(input_ids) * self.embed_scale
        else:
            student_inputs_embeds = inputs_embeds

        student_attention_mask = self._prepare_decoder_attention_mask(
            student_attention_mask, input_shape, student_inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        # Shared (for now)
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, student_inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        if self.training:
            positions = self.embed_positions(input_shape, past_key_values_length)

            hidden_states = inputs_embeds + positions
            hidden_states = self.layernorm_embedding(hidden_states)

            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        student_positions = self.student_embed_positions(input_shape, past_key_values_length)

        student_hidden_states = student_inputs_embeds + student_positions
        student_hidden_states = self.student_layernorm_embedding(student_hidden_states)

        student_hidden_states = F.dropout(student_hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        # Only output the student's outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        # Need to create and check for student_head_mask as well?
        # TODO: Need to look more into head_mask for student
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if self.training:
            student_index = 0
            teacher_hidden_states = []
            for idx, decoder_layer in enumerate(self.layers):
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (student_hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if getattr(self.config, "gradient_checkpointing", False):

                    if use_cache:
                        logger.warn(
                            "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                            "`use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, output_attentions, use_cache)

                        return custom_forward

                    if student_index < self.student_decoder_layers and self.student_layer_indices[student_index] == idx and self.config.layer_selection != 'random':
                        student_decoder_layer = self.student_layers[student_index]
                        student_layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(student_decoder_layer),
                            hidden_states,
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
                    if student_index < self.student_decoder_layers and self.student_layer_indices[student_index] == idx and self.config.layer_selection != 'random':
                        student_decoder_layer = self.student_layers[student_index]
                        student_layer_outputs = student_decoder_layer(
                            student_hidden_states,
                            attention_mask=student_attention_mask,
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

                if student_index < self.student_decoder_layers and self.student_layer_indices[student_index] == idx and self.config.layer_selection != 'random':
                    student_hidden_states = student_layer_outputs[0]
                hidden_states = layer_outputs[0]

                # Add teacher hidden states to the list
                teacher_hidden_states.append(hidden_states)

                # Interpolation
                if student_index < self.student_decoder_layers and self.student_layer_indices[student_index] == idx and self.config.layer_selection != 'random':
                    interpolation_module = self.interpolation_modules[student_index]

                    # TODO: This needs to be changed if moving from BART
                    num_teacher_layers = 12
                    # Last hidden state
                    if self.config.layer_selection == "last":
                        teacher_swap_states = teacher_hidden_states[len(teacher_hidden_states) - 1]
                    # All previous hidden states
                    elif self.config.layer_selection == "prev":
                        teacher_swap_states = random.choice(teacher_hidden_states)
                    # Disjoint Subsets
                    elif self.config.layer_selection == "disjoint":
                        ratio = int(num_teacher_layers / self.config.student_decoder_layers)
                        teacher_swap_states = random.choice(teacher_hidden_states[int(max(0, len(teacher_hidden_states) - ratio)):])

                    interpolation_output = interpolation_module(student_in=student_hidden_states, teacher_in=teacher_swap_states)
                    student_hidden_states = interpolation_output
                    hidden_states = interpolation_output

                    if use_cache:
                        next_decoder_cache += (student_layer_outputs[3 if output_attentions else 1],)

                    if output_attentions:
                        all_self_attns += (student_layer_outputs[1],)

                        if encoder_hidden_states is not None:
                            all_cross_attentions += (student_layer_outputs[2],)

                    # At this point, all computations for the student are done. Increase the student index by 1 and clear the list
                    student_index += 1
        else:
            # If the model is in eval mode, we only compute through the student layers to reduce unnecessary
            # computations with the teacher
            for idx, student_decoder_layer in enumerate(self.student_layers):
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (student_hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                student_layer_outputs = student_decoder_layer(
                    student_hidden_states,
                    attention_mask=student_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                student_hidden_states = student_layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (student_layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (student_layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (student_layer_outputs[2],)

        if self.config.layer_selection == 'random' and self.training:
            # Random layer selection: perform interpolation with random selection from teacher
            # TODO:
            for idx, student_decoder_layer in enumerate(self.student_layers):
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                if output_hidden_states:
                    all_hidden_states += (student_hidden_states,)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):
                    continue

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                student_layer_outputs = student_decoder_layer(
                    student_hidden_states,
                    attention_mask=student_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                student_hidden_states = student_layer_outputs[0]

                interpolation_module = self.interpolation_modules[idx]

                # Random selection
                teacher_swap_states = random.choice(teacher_hidden_states)

                interpolation_output = interpolation_module(student_in=student_hidden_states, teacher_in=teacher_swap_states)
                student_hidden_states = interpolation_output

                if use_cache:
                    next_decoder_cache += (student_layer_outputs[3 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (student_layer_outputs[1],)

                    if encoder_hidden_states is not None:
                        all_cross_attentions += (student_layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (student_hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [student_hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=student_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class InterpolationBartModel(BartModel):
    """

    """

    def __init__(self, config: DistilBartConfig):
        super().__init__(config=config)
        # Set up config variables
        self.student_encoder_layer_indices = config.student_encoder_layer_indices
        self.student_encoder_layers = config.student_encoder_layers
        self.student_decoder_layer_indices = config.student_decoder_layer_indices
        self.student_decoder_layers = config.student_decoder_layers

        # Create student's encoder and decoder
        # If the number of layers are the same between the student and the teacher, we use the normal encoder/decoder
        self.different_encoder, self.different_decoder = False, False
        if self.student_encoder_layers < config.encoder_layers:
            self.different_encoder = True
            self.encoder = InterpolationBartEncoder(config, self.shared)
        else:
            self.encoder = BartEncoder(config, self.shared)

        self.different_decoder = True
        self.decoder = InterpolationBartDecoder(config, self.shared)

    def load_weights_to_student(self):
        if self.different_encoder:
            self.encoder.load_weights_to_student()
        if self.different_decoder:
            self.decoder.load_weights_to_student()

    def freeze_weights(self, freeze_embedding: bool = True, freeze_encoder: bool = True):
        if freeze_embedding:
            for p in self.shared.parameters():
                p.requires_grad = False

        # Freeze decoder
        self.decoder.freeze_weights(freeze_embedding=freeze_embedding)

        # Freeze entire encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False


class InterpolationBartForQuestionAnswering(BartForQuestionAnswering):
    """

    """

    def __init__(self, config: DistilBartConfig):
        super().__init__(config=config)
        # Change the main model to the interpolation version, everything else stays the same
        self.model = InterpolationBartModel(config=config)

    def load_weights_to_student(self):
        """
        This method calls the load_weights_to_student of the main model
        """
        self.model.load_weights_to_student()

    def freeze_weights(self,
                       freeze_embedding: bool = True,
                       freeze_encoder: bool = True,
                       freeze_qa_head: bool = True):
        self.model.freeze_weights(freeze_embedding=freeze_embedding, freeze_encoder=freeze_encoder)
        if freeze_qa_head:
            for p in self.qa_outputs.parameters():
                p.requires_grad = False


class InterpolationBartForConditionalGeneration(BartForConditionalGeneration):
    """

    """

    def __init__(self, config: DistilBartConfig):
        super().__init__(config=config)
        # Change the main model to the interpolation version, everything else stays the same
        self.model = InterpolationBartModel(config=config)

    def load_weights_to_student(self):
        """
        This method calls the load_weights_to_student of the main model
        """
        self.model.load_weights_to_student()

    def freeze_weights(self,
                       freeze_embedding: bool = True,
                       freeze_encoder: bool = True,
                       freeze_lm_head: bool = True):
        self.model.freeze_weights(freeze_embedding=freeze_embedding, freeze_encoder=freeze_encoder)
        if freeze_lm_head:
            for p in self.lm_head.parameters():
                p.requires_grad = False

import copy
from typing import Optional

import torch
from torch import nn

import transformers
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartModel,
    BartForConditionalGeneration
)

from .configuration_distilbart import DistilBartConfig


class DistilBartEncoder(BartEncoder):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_encoder: BartEncoder,
                 embed_tokens: Optional[nn.Embedding] = None
                 ):
        super().__init__(config=config, embed_tokens=embed_tokens)
        # Copy embedding tokens + embedding positions + layer-norm embeeding
        self.embed_tokens.load_state_dict(bart_encoder.embed_tokens.state_dict())
        self.embed_positions.load_state_dict(bart_encoder.embed_positions.state_dict())
        self.layernorm_embedding.load_state_dict(bart_encoder.layernorm_embedding.state_dict())

        # Copy layers
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
        # Copy embedding tokens + embedding positions + layer-norm embeeding
        self.embed_tokens.load_state_dict(bart_decoder.embed_tokens.state_dict())
        self.embed_positions.load_state_dict(bart_decoder.embed_positions.state_dict())
        self.layernorm_embedding.load_state_dict(bart_decoder.layernorm_embedding.state_dict())

        # Copy layers
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
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # Encoder + Decoder
        self.encoder = DistilBartEncoder(config=config, bart_encoder=bart_model.encoder, embed_tokens=self.shared)
        self.decoder = DistilBartDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)

        # Copy pre-trained embeddings
        self.set_input_embeddings(bart_model.get_input_embeddings())


class DistilBartForConditionalGeneration(BartForConditionalGeneration):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_model_conditional: BartForConditionalGeneration):
        super().__init__(config)
        self.model = DistilBart(config=config, bart_model=bart_model_conditional.model)

        # Copy LM head
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.set_output_embeddings(bart_model_conditional.get_output_embeddings())

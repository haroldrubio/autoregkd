import copy
from typing import Optional

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
        self.layers = nn.ModuleList()
        for i in config.encoder_layer_indices:
            self.layers.append(copy.deepcopy(bart_encoder.layers[i]))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        super().forward(input_ids=input_ids,
                        attention_mask=attention_mask,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict)


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


class DistilBart(BartForConditionalGeneration):
    """
    """
    def __init__(self,
                 config: DistilBartConfig,
                 bart_model: BartModel
                 ):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = DistilBartEncoder(config=config, bart_encoder=bart_model.encoder, embed_tokens=self.shared)
        self.decoder = DistilBartDecoder(config=config, bart_decoder=bart_model.decoder, embed_tokens=self.shared)

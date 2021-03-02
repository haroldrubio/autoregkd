import copy
from typing import Optional

from torch import nn

import transformers
from transformers import BartModel
from transformers.models.bart.modeling_bart import (
    BartEncoder,
    BartDecoder,
    BartModel
)

from transformers.models.bart.configuration_bart import BartConfig
from configuration_distilbart import DistilBartConfig


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
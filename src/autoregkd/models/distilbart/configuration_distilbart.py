import transformers
from transformers.models.bart.configuration_bart import BartConfig


class DistilBartConfig(BartConfig):
    def __init__(self,
                 encoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 decoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 add_bias_logits=False,
                 add_final_layer_norm=False,
                 vocab_size=50264,
                 **kwargs
                 ):
        super().__init__(
            vocab_size=vocab_size,
            **kwargs
        )
        # Which layer(s) to copy
        self.encoder_layer_indices = encoder_layer_indices
        self.decoder_layer_indices = decoder_layer_indices
        self.encoder_layers = len(self.encoder_layer_indices)
        self.decoder_layers = len(self.decoder_layer_indices)

        # Model params
        self.add_bias_logits = add_bias_logits
        self.add_final_layer_norm = add_final_layer_norm

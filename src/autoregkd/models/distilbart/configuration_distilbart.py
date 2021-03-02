import transformers
from transformers.models.bart.configuration_bart import BartConfig


class DistilBartConfig(BartConfig):
    def __init__(self,
                 encoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 decoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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

        self.encoder_layer_indices = encoder_layer_indices
        self.decoder_layer_indices = decoder_layer_indices
        self.encoder_layers = len(self.encoder_layer_indices)
        self.decoder_layers = len(self.decoder_layer_indices)
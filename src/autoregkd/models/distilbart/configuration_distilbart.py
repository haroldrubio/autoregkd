import transformers
from transformers.models.bart.configuration_bart import BartConfig


class DistilBartConfig(BartConfig):
    def __init__(self,
                 student_encoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 student_decoder_layer_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 interpolation_type="stochastic",
                 learnable_p=False,
                 interpolation_p=0.0,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # Which layer(s) to copy
        # Note: For DistilBart, encoder_layers and student_encoder_layers are the same
        # Similarly, decoder_layers and student_decoder_layers are the same
        # For interpolation models, the pairs of variables have different values (unless the student keeps all layers)
        self.student_encoder_layer_indices = student_encoder_layer_indices
        self.student_decoder_layer_indices = student_decoder_layer_indices
        self.student_encoder_layers = len(self.student_encoder_layer_indices)
        self.student_decoder_layers = len(self.student_decoder_layer_indices)

        # Type of interpolation
        self.interpolation_type = interpolation_type
        # Whether to make p learnable
        self.learnable_p = learnable_p

        # Starting interpolation probability. This is only used by interpolation models
        self.interpolation_p = interpolation_p

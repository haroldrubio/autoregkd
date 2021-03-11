import torch
from transformers import BartForConditionalGeneration

from autoregkd.models.distilbart.modeling_distilbart import DistilBartForConditionalGeneration
from autoregkd.models.distilbart.configuration_distilbart import DistilBartConfig


def test_full_model_same_weights():
    distilbart_config = DistilBartConfig(
        encoder_layer_indices=range(12),
        decoder_layer_indices=range(12)
    )

    bart_model = BartForConditionalGeneration('facebook/bart-large-xsum')
    distilbart_model = DistilBartForConditionalGeneration(config=distilbart_config, bart_model_conditional=bart_model)

    for p1, p2 in zip(bart_model.parameters(), distilbart_model.parameters()):
        assert p1.data.ne(p2.data).sum() == 0

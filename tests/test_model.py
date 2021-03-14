import torch
from transformers import BartForConditionalGeneration

from autoregkd.models.distilbart.modeling_distilbart import create_new_student, copy_to_student
from autoregkd.models.distilbart.configuration_distilbart import DistilBartConfig


# BART model
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
bart_config_dict = bart_model.config.to_diff_dict()

# DistilBART model
distilbart_config = DistilBartConfig(
    encoder_layer_indices=range(12),
    decoder_layer_indices=range(12),
    **bart_config_dict
)
distilbart_model = create_new_student(teacher_model=bart_model,
                                      config=distilbart_config)
distilbart_config_dict = distilbart_model.config.to_diff_dict()

# Copy state_dict (weights + buffers)
copy_to_student(teacher_model=bart_model,
                student_model=distilbart_model,
                config=distilbart_config)


def test_full_model_same_weights():
    for p1, p2 in zip(bart_model.parameters(), distilbart_model.parameters()):
        assert p1.data.ne(p2.data).sum() == 0


def test_full_model_same_state_dict():
    """
    Source: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212
    """
    for ((k_1, v_1), (k_2, v_2)) in zip(bart_model.state_dict().items(), distilbart_model.state_dict().items()):
        # Assert key match
        assert k_1 == k_2

        # Convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        assert torch.allclose(v_1, v_2), "Tensor mismatch: {} and {}".format(v_1, v_2)


def test_full_model_same_config():
    for k, v in bart_config_dict.items():
        assert v == distilbart_config_dict[k], "Config mismatch key: {}".format(k)

import copy
from typing import Optional

import torch
from torch import nn

import transformers
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel
)

from .configuration_distilbart import DistilBartConfig


def create_new_student(teacher_model: BartPretrainedModel,
                       config: DistilBartConfig):
    # Get model class of the teacher
    teacher_model_class = type(teacher_model)

    # Create a student model of the same class with the given config
    student_model = teacher_model_class(config=config)
    return student_model


def copy_to_student(teacher_model: BartPretrainedModel,
                    student_model: BartPretrainedModel,
                    config: DistilBartConfig):
    student_model.load_state_dict(teacher_model.state_dict(), strict=False)

    # Copy the encoder's weights
    for i, layer_idx in enumerate(config.encoder_layer_indices):
        student_model.model.encoder.layers[i].load_state_dict(teacher_model.model.encoder.layers[layer_idx].state_dict())

    # Copy the decoder's weights
    for i, layer_idx in enumerate(config.decoder_layer_indices):
        student_model.model.decoder.layers[i].load_state_dict(teacher_model.model.decoder.layers[layer_idx].state_dict())


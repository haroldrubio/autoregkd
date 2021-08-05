# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

from ..models.interpolation.modeling_interpolation import InterpolationScheduler, LinearInterpolationModule
from .utils import InterpolationCallback


if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class SequenceInterpolationTrainer(Trainer):
    def __init__(self,
                 *args,
                 num_interpolation_epochs: int,
                 learnable_p: bool = False,
                 alpha_p: float = None,
                 max_prob: int = None,
                 per_level_annealing_duration: float = None,
                 step_size: int = None,
                 interpolation_scheduler: InterpolationScheduler = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_interpolation_epochs = num_interpolation_epochs
        self.learnable_p = learnable_p
        self.alpha_p = alpha_p
        self.max_prob = max_prob
        self.per_level_annealing_duration = per_level_annealing_duration
        self.step_size = step_size

        # Create interpolation and its callback
        self.interpolation_scheduler = interpolation_scheduler
        if self.interpolation_scheduler is not None:
            scheduler_callback = InterpolationCallback(self.interpolation_scheduler)
            self.add_callback(scheduler_callback)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)

        if not self.learnable_p:
            # Add interpolation scheduler
            if self.interpolation_scheduler is None:
                num_interpolation_steps = math.ceil(num_training_steps * self.num_interpolation_epochs / self.args.num_train_epochs)
                self.interpolation_scheduler = InterpolationScheduler(
                    interpolation_modules=self.model.model.decoder.interpolation_modules,
                    num_interpolation_steps=num_interpolation_steps,
                    max_prob=self.max_prob,
                    per_level_annealing_duration=self.per_level_annealing_duration,
                    step_size=self.step_size
                )

                scheduler_callback = InterpolationCallback(self.interpolation_scheduler)
                self.add_callback(scheduler_callback)

    def log(self, logs: Dict[str, float]) -> None:
        for i, module in enumerate(self.model.model.decoder.interpolation_modules):
            logs["decoder_p_{}".format(i)] = module.p.item()

        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.learnable_p:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

            # p regularization to encourage p to be high
            if isinstance(model, nn.DataParallel):
                interpolation_modules = model.module.model.decoder.interpolation_modules
            else:
                interpolation_modules = model.model.decoder.interpolation_modules

            for p in interpolation_modules.parameters():
                loss -= self.alpha_p * p.item()

            return (loss, outputs) if return_outputs else loss

        else:
            return super().compute_loss(model, inputs, return_outputs)

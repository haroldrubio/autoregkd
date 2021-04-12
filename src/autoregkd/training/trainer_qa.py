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


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        start_logits, end_logits = None, None
        if self.post_process_function is not None and self.compute_metrics is not None:
            start_logits = output.predictions[0]
            end_logits = output.predictions[1]
            logits = (start_logits, end_logits)
            eval_preds = self.post_process_function(eval_examples, eval_dataset, logits)
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        self._memory_tracker.start()
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        start_logits = output.predictions[0]
        end_logits = output.predictions[1]
        eval_preds = self.post_process_function(test_examples, test_dataset, (start_logits, end_logits))
        metrics = self.compute_metrics(eval_preds)

        self._memory_tracker.stop_and_update_metrics(metrics)
        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)


class QuestionAnsweringKDTrainer(QuestionAnsweringTrainer):
    def __init__(self,
                 *args,
                 use_kd_loss=False,
                 teacher_model=None,
                 temperature=2.0,
                 normalize_hidden=False,
                 alpha_data=1.0,
                 alpha_logits=0.0,
                 alpha_hidden=0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_kd_loss = use_kd_loss
        self.teacher_model = teacher_model
        # Assert teacher model is in eval mode
        if self.teacher_model is not None:
            assert not self.teacher_model.training

        # Get the configurations to compare sizes
        self.student_config_dict = self.model.config.to_diff_dict()
        if self.teacher_model is not None:
            self.teacher_config_dict = self.teacher_model.config.to_diff_dict()

        self.temperature = temperature
        self.normalize_hidden = normalize_hidden

        self.alpha_data = alpha_data
        self.alpha_logits = alpha_logits
        self.alpha_hidden = alpha_hidden

    def compute_loss(self, model, inputs, return_outputs=False):
        # Update inputs to output hidden states and in form of a dictionary
        inputs["output_hidden_states"] = self.use_kd_loss
        inputs["return_dict"] = True

        # Compute data loss, which is the default loss of the model
        data_loss, student_outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Compute KD component losses
        # Initialize losses to all 0s and only update if we use knowledge-distillation loss
        enc_hidden_loss, dec_hidden_loss, logits_loss = 0.0, 0.0, 0.0
        if self.use_kd_loss and self.teacher_model is not None:
            # Set up variables
            input_ids, source_mask = inputs["input_ids"], inputs["attention_mask"]
            pad_token_id = self.tokenizer.pad_token_id
            decoder_input_ids = shift_tokens_right(input_ids=input_ids,
                                                   pad_token_id=pad_token_id,
                                                   decoder_start_token_id=self.teacher_model.config.decoder_start_token_id)

            teacher_model = self.teacher_model.to(input_ids.device)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids,
                                                attention_mask=source_mask,
                                                decoder_input_ids=decoder_input_ids,
                                                output_hidden_states=True,
                                                return_dict=True,
                                                use_cache=False)

            # Compute logits loss
            student_logits_combined = torch.stack([student_outputs.start_logits, student_outputs.end_logits], dim=1)
            teacher_logits_combined = torch.stack([teacher_outputs.start_logits, teacher_outputs.end_logits], dim=1)
            logits_loss = self._compute_logits_loss(student_logits=student_logits_combined,
                                                    teacher_logits=teacher_logits_combined,
                                                    temperature=self.temperature)

            # Only compute encoder's hidden loss if the student's encoder is smaller
            if self.student_config_dict["encoder_layers"] < self.teacher_config_dict["encoder_layers"]:
                enc_hidden_loss = self._compute_hidden_loss(
                    student_hidden_states=student_outputs.encoder_hidden_states,
                    teacher_hidden_states=teacher_outputs.encoder_hidden_states,
                    attention_mask=source_mask,
                    teacher_layer_indices=self.student_config_dict["encoder_layer_indices"],
                    normalize=self.normalize_hidden
                )

            # Only compute decoder's hidden loss if the student's decoder is smaller
            if self.student_config_dict["decoder_layers"] < self.teacher_config_dict["decoder_layers"]:
                decoder_mask = decoder_input_ids.ne(pad_token_id)
                dec_hidden_loss = self._compute_hidden_loss(
                    student_hidden_states=student_outputs.decoder_hidden_states,
                    teacher_hidden_states=teacher_outputs.decoder_hidden_states,
                    attention_mask=decoder_mask,
                    teacher_layer_indices=self.student_config_dict["decoder_layer_indices"],
                    normalize=self.normalize_hidden
                )

        total_loss = (self.alpha_data * data_loss
                      + self.alpha_logits * logits_loss
                      + self.alpha_hidden * (enc_hidden_loss + dec_hidden_loss))

        return (total_loss, student_outputs) if return_outputs else total_loss

    @staticmethod
    def _compute_logits_loss(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             temperature: float = 2.0):
        assert (
                student_logits.shape == teacher_logits.shape
        ), "Expected tensors of the same size. Got student: {}, teacher: {}".format(student_logits.shape,
                                                                                    teacher_logits.shape)
        assert (
                student_logits.shape[1] == teacher_logits.shape[1] == 2
        ), "Expected 2 for dim 1 of logit tensors (start + end logits). Got: {}".format(student_logits.shape[1])

        # Compute logits loss
        logits_loss_fct = nn.KLDivLoss(reduction="batchmean")
        logits_loss = (
                logits_loss_fct(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.log_softmax(teacher_logits / temperature, dim=-1)
                ) * temperature ** 2
        )

        return logits_loss

    @staticmethod
    def _compute_hidden_loss(student_hidden_states: Tuple[torch.Tensor],
                             teacher_hidden_states: Tuple[torch.Tensor],
                             attention_mask: torch.Tensor,
                             teacher_layer_indices: list,
                             normalize: bool = False
                             ):
        mask = attention_mask.to(student_hidden_states[0])  # Type and/or device conversion
        valid_count = mask.sum() * student_hidden_states[0].size(-1)  # Get valid count

        # Stack hidden states
        # Here we skip the first hidden state which is the output of the embeddings
        student_hidden_stack = torch.stack([state for state in student_hidden_states[1:]])
        teacher_hidden_stack = torch.stack([teacher_hidden_states[i] for i in teacher_layer_indices])
        assert (
                student_hidden_stack.shape == teacher_hidden_stack.shape
        ), "Expected tensors of the same size. Got student: {}, teacher: {}".format(student_hidden_stack.shape,
                                                                                    teacher_hidden_stack.shape)

        # Normalize if specified
        if normalize:
            student_hidden_stack = F.layer_norm(student_hidden_stack, student_hidden_stack.shape[1:])
            teacher_hidden_stack = F.layer_norm(teacher_hidden_stack, teacher_hidden_stack.shape[1:])

        # Compute MSE loss
        loss_fct = nn.MSELoss(reduction="none")
        mse_loss = loss_fct(student_hidden_stack, teacher_hidden_stack)
        masked_mse_loss = (mse_loss * mask.unsqueeze(dim=0).unsqueeze(dim=-1)).sum() / valid_count

        return masked_mse_loss


class QuestionAnsweringInterpolationTrainer(QuestionAnsweringTrainer):
    def __init__(self,
                 *args,
                 num_interpolation_epochs: int,
                 max_prob: int,
                 per_level_annealing_duration: float,
                 step_size: int,
                 interpolation_scheduler: InterpolationScheduler = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.num_interpolation_epochs = num_interpolation_epochs
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
        if self.interpolation_scheduler is not None:
            for i, module in enumerate(self.model.model.decoder.interpolation_modules):
                logs["decoder_p_{}".format(i)] = module.p.item()

        super().log(logs)


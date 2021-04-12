"""
Seq2Seq trainer with Knowledge-Distillation (KD) loss
Based on https://github.com/huggingface/transformers/blob/master/examples/research_projects/seq2seq-distillation/distillation.py
"""
import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F

import transformers
from transformers import Seq2SeqTrainer
from transformers.models.bart.modeling_bart import shift_tokens_right

from ..models.interpolation.modeling_interpolation import InterpolationScheduler
from .utils import InterpolationCallback


class Seq2SeqKDTrainer(Seq2SeqTrainer):
    """

    """

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

        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)

        self.temperature = temperature
        self.normalize_hidden = normalize_hidden

        self.alpha_data = alpha_data
        self.alpha_logits = alpha_logits
        self.alpha_hidden = alpha_hidden

    def compute_loss(self, model, inputs, return_outputs=False):
        # Update inputs to output hidden states and in form of a dictionary
        inputs["output_hidden_states"] = self.use_kd_loss
        inputs["return_dict"] = True

        # Compute cross-entropy data loss, which is identical to the default loss of Seq2SeqTrainer
        data_loss, student_outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Compute KD component losses
        # Initialize losses to all 0s and only update if we use knowledge-distillation loss
        enc_hidden_loss, dec_hidden_loss, logits_loss = 0.0, 0.0, 0.0
        if self.use_kd_loss:
            # Set up variables
            input_ids, source_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
            pad_token_id = self.tokenizer.pad_token_id
            decoder_input_ids = shift_tokens_right(input_ids=labels,
                                                   pad_token_id=pad_token_id,
                                                   decoder_start_token_id=self.student_config_dict["decoder_start_token_id"])

            assert (
                    self.teacher_model.device == input_ids.device
            ), "Expected teacher model and input IDs to be on the same device. Got teacher model on {} and input IDs on {}".format(
                self.teacher_model.device, input_ids.device)

            with torch.no_grad():
                teacher_outputs = self.teacher_model(input_ids=input_ids,
                                                     attention_mask=source_mask,
                                                     decoder_input_ids=decoder_input_ids,
                                                     output_hidden_states=True,
                                                     return_dict=True,
                                                     use_cache=False)

                # Compute logits loss
            decoder_mask = decoder_input_ids.ne(pad_token_id)
            logits_loss = self._compute_logits_loss(student_logits=student_outputs.logits,
                                                    teacher_logits=teacher_outputs.logits,
                                                    mask=decoder_mask,
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
                dec_hidden_loss = self._compute_hidden_loss(
                    student_hidden_states=student_outputs.decoder_hidden_states,
                    teacher_hidden_states=teacher_outputs.decoder_hidden_states,
                    attention_mask=decoder_mask,
                    teacher_layer_indices=self.student_config_dict["decoder_layer_indices"],
                    normalize=self.normalize_hidden
                )

        total_loss = self.alpha_data * data_loss + \
                     self.alpha_logits * logits_loss + \
                     self.alpha_hidden * (enc_hidden_loss + dec_hidden_loss)

        return (total_loss, student_outputs) if return_outputs else total_loss

    @staticmethod
    def _compute_logits_loss(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor,
                             mask: torch.Tensor,
                             temperature: float = 2.0):
        sel_mask = mask[:, :, None].expand_as(student_logits)
        vocab_size = student_logits.size(-1)

        # Select logits based on mask
        student_logits_select = torch.masked_select(student_logits, sel_mask).view(-1, vocab_size)
        teacher_logits_select = torch.masked_select(teacher_logits, sel_mask).view(-1, vocab_size)
        assert (
                student_logits_select.shape == teacher_logits_select.shape
        ), "Expected tensors of the same size. Got student: {}, teacher: {}".format(student_logits_select.shape,
                                                                                    teacher_logits_select.shape)

        # Compute logits loss
        logits_loss_fct = nn.KLDivLoss(reduction="batchmean")
        logits_loss = (
                logits_loss_fct(
                    F.log_softmax(student_logits_select / temperature, dim=-1),
                    F.log_softmax(teacher_logits_select / temperature, dim=-1)
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


class Seq2SeqInterpolationTrainer(Seq2SeqTrainer):
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

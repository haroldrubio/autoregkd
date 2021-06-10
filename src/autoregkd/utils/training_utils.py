import warnings
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Callable

import torch
import wandb
from torch import nn
from torch.optim import Optimizer

from transformers import (
    TrainerCallback,
    TrainerState,
    Trainer,
    Seq2SeqTrainer
)

from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.optimization import Adafactor, AdamW

from ..models.custom_bart import(
    InterpolationScheduler,
    InterpolationSchedulerV2s,
    LRV2s,
    InterpolationSchedulerPLAD,
    TheseusScheduler
)

class SchedulerState(TrainerState):
    dec_interpolate_type: str = None
    model_optimizer: Optimizer = None
    prob_scheduler: InterpolationScheduler = None

class SchedulerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        state.prob_scheduler.step()
        return super().on_step_end(args, state, control, **kwargs)
    def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
        if state.dec_interpolate_type == 'warmup':
            # First, retrieve the probabilities
            probs = state.prob_scheduler.probs
            # Then, multiply through the corresponding parameter groups
            # Retrieve the first n groups
            layer_groups = optimizer.param_groups[:len(probs)]
            # Modify the learning rates
            for idx, group in enumerate(layer_groups):
                group['lr'] = optimizer.defaults['lr'] * abs(1 - probs[idx])
        return super().on_step_begin(args, state, control, **kwargs)

class DistilTrainer(Trainer):
    def __init__(self, *args, scheduler_args=None, dec_interpolate_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_args = scheduler_args
        self.prob_scheduler = None
        self.dec_interpolate_type = dec_interpolate_type
    # Harold: Hack to force state to be one that can store the scheduler
    def num_examples(self, dataloader):
        if not isinstance(self.state, SchedulerState) and self.scheduler_args is not None:
            self.state = SchedulerState()
            self.state.dec_interpolate_type = self.dec_interpolate_type
            self.state.prob_scheduler = self.prob_scheduler
        return super().num_examples(dataloader)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # TODO: create parameter groups for each student BART decoder layer
        # Each layer is stored in self.model.model.decoder.layers
        # Name: "model.decoder.layers.i" for layer i

        # For all named parameters: bin them into decoder layers, and every other ones
        # Re-instantiate the optimizer - Replace super call with direct code and modify where necessary

        # Check if performing interpolation by checking scheduler_args
        if self.scheduler_args is not None:
            # Fetch interpolation modules and create scheduler
            if self.dec_interpolate_type != 'warmup':
                modules = self.model.model.decoder.interp
            else:
                modules = self.model.model.decoder.layers

            if self.dec_interpolate_type == 'interpolate':
                self.prob_scheduler = InterpolationScheduler(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'interpolatev2s' or 'attention' in self.dec_interpolate_type or self.dec_interpolate_type == 'distribution':
                # PLAD: switch to PLAD scheduling
                self.prob_scheduler = InterpolationSchedulerV2s(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'plad':
                self.prob_scheduler = InterpolationSchedulerPLAD(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'theseus':
                self.prob_scheduler = TheseusScheduler(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'warmup':
                # Progressively warmup the network
                # Declare a scheduler
                self.prob_scheduler = LRV2s(modules, self.scheduler_args, num_training_steps)
                decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
                decay_parameters = [name for name in decay_parameters if "bias" not in name]

                # Assemble layer-wise parameter groups
                # Only group by decoder layer - no need to group by weight decay
                params_at_layer = {i:[] for i in range(len(modules))}
                gen_params, decay_gen_params = [], []
                for n, p in self.model.named_parameters():
                    is_decoder_layer = False
                    for idx in range(len(modules)):
                        # Check if this parameter belongs to this layer
                        compare_string = f'decoder.layers.{idx}'
                        if compare_string in n:
                            is_decoder_layer = True
                            params_at_layer[idx].append(p)
                    # Handle general parameter
                    if not is_decoder_layer:
                        if n in decay_parameters:
                            decay_gen_params.append(p)
                        else:
                            gen_params.append(p)
                
                # Assemble list of parameter groups
                # Handle decoder layers
                grouped_params = []
                for idx in range(len(modules)):
                    p_list = params_at_layer[idx]
                    # Assemble dictionaries
                    p_dict = {'params': p_list, 'weight_decay': self.args.weight_decay}
                    grouped_params.append(p_dict)
                    
                # Handle general parameters
                p_dict = {'params': gen_params, 'weight_decay': 0.0}
                decay_p_dict = {'params': decay_gen_params, 'weight_decay': self.args.weight_decay}
                grouped_params.append(p_dict)
                grouped_params.append(decay_p_dict)

                # Replicate HF code
                optimizer_cls = Adafactor if self.args.adafactor else AdamW
                if self.args.adafactor:
                    optimizer_cls = Adafactor
                    optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
                else:
                    optimizer_cls = AdamW
                    optimizer_kwargs = {
                        "betas": (self.args.adam_beta1, self.args.adam_beta2),
                        "eps": self.args.adam_epsilon,
                    }
                optimizer_kwargs["lr"] = self.args.learning_rate
                # Harold: this approach does not support distributed training
                self.optimizer = optimizer_cls(grouped_params, **optimizer_kwargs)

            # Overwrite state
            self.state = SchedulerState()
            self.state.prob_scheduler = self.prob_scheduler
    
    def log(self, logs):
        if self.scheduler_args is not None and self.dec_interpolate_type != 'warmup':
            # Record swapping probabilities
            modules = self.model.model.decoder.interp
            for idx, mod in enumerate(modules):
                logs[f'swap_prob_{idx}'] = float(mod.swap_prob)
        if self.dec_interpolate_type == 'warmup':
            # Record the scheduler itself
            probs = self.state.prob_scheduler.probs
            for idx, prob in enumerate(probs):
                logs[f'lr_weight_at_layer_{idx}'] = float(abs(1 - prob))
            # Record parameter groups
            for idx, group in enumerate(self.optimizer.param_groups):
                logs[f'lr_at_layer_{idx}'] = group['lr']
        if 'attention' in self.dec_interpolate_type and self.model.training:
            attn_list = self.model.attention_list
            for idx, attn_scores in enumerate(attn_list):
                # Convert tensor to numpy
                print(attn_scores.shape)
                cpu_scores = attn_scores.detach().cpu().numpy()
                logs[f'attn_at_std_layer_{idx}'] = wandb.Histogram(cpu_scores)
        super().log(logs)

# ---------------------- Replicate for Seq2Seq Trainer ----------------------
class DistilSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, scheduler_args=None, dec_interpolate_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_args = scheduler_args
        self.prob_scheduler = None
        self.dec_interpolate_type = dec_interpolate_type
    # Harold: Hack to force state to be one that can store the scheduler
    def num_examples(self, dataloader):
        if not isinstance(self.state, SchedulerState) and self.scheduler_args is not None:
            self.state = SchedulerState()
            self.state.dec_interpolate_type = self.dec_interpolate_type
            self.state.prob_scheduler = self.prob_scheduler
        return super().num_examples(dataloader)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # TODO: create parameter groups for each student BART decoder layer
        # Each layer is stored in self.model.model.decoder.layers
        # Name: "model.decoder.layers.i" for layer i

        # For all named parameters: bin them into decoder layers, and every other ones
        # Re-instantiate the optimizer - Replace super call with direct code and modify where necessary

        # Check if performing interpolation by checking scheduler_args
        if self.scheduler_args is not None:
            # Fetch interpolation modules and create scheduler
            if self.dec_interpolate_type != 'warmup':
                modules = self.model.model.decoder.interp
            else:
                modules = self.model.model.decoder.layers

            if self.dec_interpolate_type == 'interpolate':
                self.prob_scheduler = InterpolationScheduler(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'interpolatev2s' or 'attention' in self.dec_interpolate_type:
                # PLAD: switch to PLAD scheduling
                self.prob_scheduler = InterpolationSchedulerV2s(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'plad':
                self.prob_scheduler = InterpolationSchedulerPLAD(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'theseus':
                self.prob_scheduler = TheseusScheduler(modules, self.scheduler_args, num_training_steps)

            elif self.dec_interpolate_type == 'warmup':
                # Progressively warmup the network
                # Declare a scheduler
                self.prob_scheduler = LRV2s(modules, self.scheduler_args, num_training_steps)
                decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
                decay_parameters = [name for name in decay_parameters if "bias" not in name]

                # Assemble layer-wise parameter groups
                # Only group by decoder layer - no need to group by weight decay
                params_at_layer = {i:[] for i in range(len(modules))}
                gen_params, decay_gen_params = [], []
                for n, p in self.model.named_parameters():
                    is_decoder_layer = False
                    for idx in range(len(modules)):
                        # Check if this parameter belongs to this layer
                        compare_string = f'decoder.layers.{idx}'
                        if compare_string in n:
                            is_decoder_layer = True
                            params_at_layer[idx].append(p)
                    # Handle general parameter
                    if not is_decoder_layer:
                        if n in decay_parameters:
                            decay_gen_params.append(p)
                        else:
                            gen_params.append(p)
                
                # Assemble list of parameter groups
                # Handle decoder layers
                grouped_params = []
                for idx in range(len(modules)):
                    p_list = params_at_layer[idx]
                    # Assemble dictionaries
                    p_dict = {'params': p_list, 'weight_decay': self.args.weight_decay}
                    grouped_params.append(p_dict)
                    
                # Handle general parameters
                p_dict = {'params': gen_params, 'weight_decay': 0.0}
                decay_p_dict = {'params': decay_gen_params, 'weight_decay': self.args.weight_decay}
                grouped_params.append(p_dict)
                grouped_params.append(decay_p_dict)

                # Replicate HF code
                optimizer_cls = Adafactor if self.args.adafactor else AdamW
                if self.args.adafactor:
                    optimizer_cls = Adafactor
                    optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
                else:
                    optimizer_cls = AdamW
                    optimizer_kwargs = {
                        "betas": (self.args.adam_beta1, self.args.adam_beta2),
                        "eps": self.args.adam_epsilon,
                    }
                optimizer_kwargs["lr"] = self.args.learning_rate
                # Harold: this approach does not support distributed training
                self.optimizer = optimizer_cls(grouped_params, **optimizer_kwargs)

            # Overwrite state
            self.state = SchedulerState()
            self.state.prob_scheduler = self.prob_scheduler
    
    def log(self, logs):
        if self.scheduler_args is not None and self.dec_interpolate_type != 'warmup':
            # Record swapping probabilities
            modules = self.model.model.decoder.interp
            for idx, mod in enumerate(modules):
                logs[f'swap_prob_{idx}'] = float(mod.swap_prob)
        if self.dec_interpolate_type == 'warmup':
            # Record the scheduler itself
            probs = self.state.prob_scheduler.probs
            for idx, prob in enumerate(probs):
                logs[f'lr_weight_at_layer_{idx}'] = float(abs(1 - prob))
            # Record parameter groups
            for idx, group in enumerate(self.optimizer.param_groups):
                logs[f'lr_at_layer_{idx}'] = group['lr']
        # DEBUG: temp disable attention logging
        if 'attention' in self.dec_interpolate_type:
            attn_list = self.model.attention_list
            for idx, attn_scores in enumerate(attn_list):
                # Convert tensor to numpy
                cpu_scores = attn_scores.detach().cpu().numpy()
                logs[f'attn_at_std_layer_{idx}'] = wandb.Histogram(cpu_scores)

        super().log(logs)
import warnings
from pathlib import Path
from typing import List, Tuple, Union, Iterable, Callable

from torch import nn

from transformers import (
    TrainerCallback,
    TrainerState,
    Trainer
)

from ..models.custom_bart import(
    InterpolationScheduler,
    InterpolationSchedulerV2s,
    InterpolationSchedulerPLAD
)

class SchedulerState(TrainerState):
    prob_scheduler: InterpolationScheduler = None

class SchedulerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        state.prob_scheduler.step()
        return super().on_step_end(args, state, control, **kwargs)

class DistilTrainer(Trainer):
    def __init__(self, *args, scheduler_args=None, dec_interpolate_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler_args = scheduler_args
        self.prob_scheduler = None
        self.dec_interpolate_type = dec_interpolate_type
    # Harold: Hack to force state to be one that can store the scheduler
    def num_examples(self, dataloader):
        if not isinstance(self.state, SchedulerState):
            self.state = SchedulerState()
            self.state.prob_scheduler = self.prob_scheduler
        return super().num_examples(dataloader)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        # Check if performing interpolation by checking scheduler_args
        if self.scheduler_args is not None:
            # Fetch interpolation modules and create scheduler
            modules = self.model.model.decoder.interp
            if self.dec_interpolate_type == 'interpolate':
                self.prob_scheduler = InterpolationScheduler(modules, self.scheduler_args, num_training_steps)
            elif self.dec_interpolate_type == 'interpolatev2s':
                # PLAD: switch to PLAD scheduling
                # self.prob_scheduler = InterpolationSchedulerV2s(modules, self.scheduler_args, num_training_steps)
                self.prob_scheduler = InterpolationSchedulerPLAD(modules, self.scheduler_args, num_training_steps)
            # Overwrite state
            self.state = SchedulerState()
            self.state.prob_scheduler = self.prob_scheduler
    
    def log(self, logs):
        if self.scheduler_args is not None:
            # Record swapping probabilities
            modules = self.model.model.decoder.interp
            for idx, mod in enumerate(modules):
                logs[f'swap_prob_{idx}'] = float(mod.swap_prob)
        super().log(logs)


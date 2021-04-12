from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from ..models.interpolation.modeling_interpolation import InterpolationScheduler


class InterpolationState(TrainerState):
    interpolation_scheduler: InterpolationScheduler = None


class InterpolationCallback(TrainerCallback):
    def __init__(self, scheduler: InterpolationScheduler):
        self.scheduler = scheduler

    def on_step_end(self, args: TrainingArguments, state: InterpolationState, control: TrainerControl, **kwargs):
        self.scheduler.step()


class AddAtEpochCallback(TrainerCallback):
    """
    Add early stopping callback after a certain number of epochs
    This is to prevent the model early stopped when the loss increases as a result of a large interpolation p
    """
    def __init__(self,
                 trainer,
                 num_interpolation_epochs: int,
                 callback: TrainerCallback):
        self.trainer = trainer
        self.num_interpolation_epochs = num_interpolation_epochs
        self.callback = callback

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch == self.num_interpolation_epochs:
            self.trainer.add_callback(self.callback)

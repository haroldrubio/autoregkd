# Knowledge Distillation For Generative Autoregressive Models
## Execution
Run an experiment with `python run_experiment.py` where the command line arguments are specified in `src/autoregkd/interface/cli.py`
### Weights and Biases Logging and Sweeping
Log into W and B by running `wandb login` in the command line and pasting your API key \
The Huggingface trainer has direct logging integration with Weights and Biases

Start a sweep with `wandb sweep scripts/sweep.yaml` \
Follow the instructions to add agents to the sweep
## Modifications
### Adding a custom hyperparameter
1) Add a click option into `cli.py`
```
  @click.option(
    "--hypparam_name",
    type=str,
    default='Parameter Name',
    help="random parameter",
  )
```
2) Add the hyperparameter into the `CustomArguments` dataclass
```
hypparam_name: str = field(
    default="Hello World",
    metadata={
        "help": (
            "A simple test"
        )
    },
)
```
3) Add a keyword argument into `training_args` in `train.py`: `hypparam_name=config['hypparam_name']`
### Changing the learning process
Initialize additional schedulers into the `create_optimizer_and_scheduler` function of `CustomTrainer` and step them in the `train` function. \
Learn with a new loss by changing the `compute_loss` function.
### Creating new BART architectures
Use the encoder/decoder BART layers in `custom_bart.py` to create new encoders/decoders. Then, create a subclass of `BartModel` and override the existing encoder/decoder.


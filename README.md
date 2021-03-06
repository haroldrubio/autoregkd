# Knowledge Distillation For Generative Autoregressive Models
## Execution
Run an experiment with `python run_experiment.py` where the command line arguments are specified in `src/autoregkd/utils/custom_args.py`
and the HuggingFace [Trainer](https://huggingface.co/transformers/main_classes/trainer.html#Seq2SeqTrainingArguments) documentation
### Weights and Biases Logging and Sweeping
Log into W and B by running `wandb login` in the command line and pasting your API key \
The Huggingface trainer has direct logging integration with Weights and Biases

Start a sweep with `wandb sweep scripts/sweep.yaml` \
Follow the instructions to add agents to the sweep
## Modifications
### Adding a custom hyperparameter
1) Add the hyperparameter into one of the dataclasses in `custom_args.py`
```
hyper_param: str = field(
        default="Hello World!",
        metadata={"help": "Description of the hyperparameter"}
    )
```
2) In the corresponding trainer in `trainers.py`, override the function that your hyperparameter would be affecting.
3) The hyperparameter in the trainer is accessible by: `self.args.hyper_param`
### Changing the learning process
Initialize additional schedulers into the `create_optimizer_and_scheduler` function of the corresponding trainer and step them in the `train` function. \
Learn with a new loss by changing the `compute_loss` function.
### Creating new BART architectures
Use the encoder/decoder BART layers in `custom_bart.py` to create new encoders/decoders. Then, drop in the decoder/encoder into a BART model (BartForConditionalGeneration, BartForQuestionAnswering, etc.)


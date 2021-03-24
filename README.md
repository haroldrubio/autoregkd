# Knowledge Distillation For Generative Autoregressive Models
## Execution
Run an experiment with `python run_experiment.py` where the command line arguments are specified in `src/autoregkd/interface/cli.py`
### Weights and Biases Logging and Sweeping
Log into W and B by running `wandb login` in the command line and pasting your API key \
The Huggingface trainer has direct logging integration with Weights and Biases

Start a sweep with `wandb sweep scripts/sweep.yaml` \
Follow the instructions to add agents to the sweep
## Interpolation Decoder
You should be able to drop in the `InterpolationDecoder` directly into your model's BART decoder. You can load in all the weights from the teacher model directly into the `InterpolationDecoder` as the teacher layers should exist unchanged - there is simply an extra copy for the student network prefixed with `std_`.

**IMPORTANT**: After the decoder's teacher network weights have been loaded, make sure to call `decoder.setup_interpolation()`. This performs the following steps:
1) Loads the teacher embeddings into the student embeddings
2) Freezes the student network embeddings
3) Freezes the teacher network decoder layers
4) Unfreezes the student network decoder layers

In order to additionally compute a loss across the teacher logits, you should create a subclass of `BartFor$[TASK]` that overrides the `forward()` function, as these models compute the loss in the forward pass. See `DistilBartForQuestionAnswering` for an example

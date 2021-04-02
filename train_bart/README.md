# Fine-Tuning BART for Question Answering
Requires: Python 3.8, Cuda 11 (for PyTorch)

The command line arguments are specified in `args.json` and a run can be started with:
```
python run_qa.py args.json
```
For extra reference on acceptable arguments, see the `TrainerArguments` documentation [here](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments),
along with the first few lines of the `run_qa.py` file, which contains task-specific arugments.

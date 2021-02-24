
from torch.utils.data.dataset import Dataset
from transformers import BartForSequenceClassification
from sklearn.metrics import accuracy_score
from typing import *
import wandb
from ..utils.custom_args import CustomArguments
from ..utils.custom_trainer import CustomTrainer

def compute_metrics(pred):
    labels = pred.label_ids
    p1, _ = pred.predictions
    preds = p1.argmax(1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def training(config: Dict) -> None:
    """
    config: dict - Contains arguments in the values whose keys are the arguments specified in interface/cli.py
    """

    # Load data
    train_data = Dataset()
    test_data = Dataset()

    # Load model
    model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=2)

    # Build directory
    output_dir = f"/runs/results_{config['learning_rate']:.2e}"
    wandb.init(project='autoregkd', name=output_dir)

    training_args = CustomArguments(
        output_dir=output_dir,          # output directory
        num_train_epochs=config['epochs'],              # total number of training epochs
        per_device_train_batch_size=2**config['log_batch_size'],  # batch size per device during training
        per_device_eval_batch_size=2**config['log_eval_batch_size'],   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        learning_rate=config['learning_rate'], # learning rate
        evaluation_strategy='steps',
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )

    trainer = CustomTrainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_data,            # training dataset
        eval_dataset=test_data,               # evaluation dataset
        compute_metrics=compute_metrics
    )    

    trainer.train()
    trainer.evaluate()

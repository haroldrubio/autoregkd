import logging
from filelock import FileLock

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    BartModel,
    Seq2SeqTrainer,
    set_seed
)
from ..models.custom_bart import DistilBart, DistilBartConfig
from ..utils.dataset import HF_Dataset



logger = logging.getLogger(__name__)

def training(model_args, data_args, training_args) -> None:
    # Set seed for replicability
    set_seed(training_args.seed)

    # Load dataset
    data_accessor = HF_Dataset(model_args, data_args)
    train_dataset, val_dataset, test_dataset, data_collator = data_accessor.access_datasets()

    # DistilBART configuration
    config = DistilBartConfig.from_pretrained(model_args.model_name)
    config.set_distillation(list(model_args.encoder_layer_indices), list(model_args.decoder_layer_indices))

    # BART tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)

    # DistilBART model
    # TODO: Change this depending on model_args
    bart_model = BartModel.from_pretrained(model_args.model_name)
    distilbart_model = DistilBart(config=config, bart_model=bart_model)
    gen_model = BartForConditionalGeneration.from_pretrained(model_args.model_name)
    gen_model.model = distilbart_model

    # Trainer
    trainer = Seq2SeqTrainer(
        model=gen_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=data_accessor.compute_metrics
    )

    # Training
    logging.info("*** Training ***")
    train_result = trainer.train(resume_from_checkpoint=None)
    # trainer.save_model()

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #trainer.log_metrics("train", metrics)
    #trainer.save_metrics("train", metrics)
    #trainer.save_state()

    # Evaluation
    if val_dataset:
        logging.info("*** Evaluating ***")
        results = {}
        metrics = trainer.evaluate(max_length=data_args.max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval")
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(val_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(val_dataset))

    #trainer.log_metrics("eval", metrics)
    #trainer.save_metrics("eval", metrics)

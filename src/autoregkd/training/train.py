import logging
from filelock import FileLock

from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartModel,
    Seq2SeqTrainer,
    set_seed
)
from .trainers import QuestionAnsweringTrainer
from ..models.custom_bart import DistilBart, DistilBartConfig
from ..utils.dataset import Gen_Dataset, QA_Dataset

logger = logging.getLogger(__name__)

def training(model_args, data_args, training_args) -> None:
    # Set seed for replicability
    set_seed(training_args.seed)

    # DistilBART configuration
    config = DistilBartConfig.from_pretrained(model_args.model_name)
    config.set_distillation(list(model_args.encoder_layer_indices), list(model_args.decoder_layer_indices))
    bart_model = BartModel.from_pretrained(model_args.model_name)
    distilbart_model = DistilBart(config=config, bart_model=bart_model)

    # Load dataset
    if data_args.task == 'summarization':
        data_accessor = Gen_Dataset(model_args, data_args)
        train_dataset, val_dataset, test_dataset, data_collator = data_accessor.access_datasets()
        curr_model = BartForConditionalGeneration.from_pretrained(model_args.model_name)
        tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)
    elif data_args.task == 'question-answering':
        data_accessor = QA_Dataset(training_args, model_args, data_args)
        train_dataset, val_dataset, data_collator = data_accessor.access_datasets()
        curr_model = BartForQuestionAnswering.from_pretrained(model_args.model_name)
        tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name)

    curr_model.model = distilbart_model

    # Trainer
    if data_args.task == 'summarization':
        trainer = Seq2SeqTrainer(
            model=curr_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=data_accessor.compute_metrics
        )
    elif data_args.task == 'question-answering':
        trainer = QuestionAnsweringTrainer(
            model=curr_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            eval_examples=data_accessor.raw_val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=data_accessor.postprocess_text,
            compute_metrics=data_accessor.compute_metrics,
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

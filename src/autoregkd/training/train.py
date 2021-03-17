import logging
from filelock import FileLock
import sys
from transformers import (
    AutoConfig,
    ElectraTokenizerFast,
    ElectraForQuestionAnswering,
    BartConfig,
    BartTokenizer,
    BartTokenizerFast,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartModel,
    Seq2SeqTrainer,
    set_seed
)
from .trainers import QuestionAnsweringTrainer, Seq2SeqTrainer
from ..models.custom_bart import DistilBart, DistilBartConfig
from ..utils.dataset import Gen_Dataset, QA_Dataset

logger = logging.getLogger(__name__)

def training(model_args, data_args, training_args) -> None:
    # Set seed for replicability
    set_seed(training_args.seed)

    # DistilBART configuration
    '''
    bart_config = BartConfig.from_pretrained(model_args.model_name).to_diff_dict()
    config = DistilBartConfig(encoder_layer_indices=list(model_args.encoder_layer_indices),
                              decoder_layer_indices=list(model_args.decoder_layer_indices),
                              model_name=model_args.model_name,
                              swap_prob=model_args.swap_prob,
                              decoder_type=training_args.model_type,
                              **bart_config)
    '''
    config = DistilBartConfig().from_pretrained(model_args.model_name)
    config.set_distillation(list(model_args.encoder_layer_indices), list(model_args.decoder_layer_indices))
    config.decoder_type = training_args.model_type
    bart_model = BartModel.from_pretrained(model_args.model_name)
    distilbart_model = DistilBart(config=config, bart_model=bart_model)
    # Load dataset
    curr_model = None
    if data_args.task == 'summarization':
        data_accessor = Gen_Dataset(model_args, data_args)
        train_dataset, val_dataset, test_dataset, data_collator = data_accessor.access_datasets()
        curr_model = BartForConditionalGeneration.from_pretrained(model_args.model_name)
        tokenizer = BartTokenizer.from_pretrained(model_args.tokenizer_name)
    elif data_args.task == 'question-answering':
        config = AutoConfig.from_pretrained(
            model_args.model_name,
        )
        data_accessor = QA_Dataset(training_args, model_args, data_args)
        train_dataset, val_dataset, data_collator = data_accessor.access_datasets()
        curr_model = BartForQuestionAnswering.from_pretrained(model_args.model_name, config=config)
        tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name)
    else:
        raise ValueError("Unsupported task")
    
    # DEBUG - disable distilled model
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

    
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=None)
    if training_args.do_eval:
        trainer.evaluate(val_dataset)
    if model_args.save_final:
        trainer.save_model()
    
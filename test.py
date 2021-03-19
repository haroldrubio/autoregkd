from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from autoregkd.models.distilbart.configuration_distilbart import DistilBartConfig
from autoregkd.models.distilbart.modeling_distilbart import *


# BART model
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')
bart_config_dict = bart_model.config.to_diff_dict()

# DistilBART model
distilbart_config = DistilBartConfig(
    encoder_layer_indices=range(12),
    decoder_layer_indices=range(12),
    **bart_config_dict
)
distilbart_model = create_new_student(teacher_model=bart_model,
                                      config=distilbart_config)
distilbart_config_dict = distilbart_model.config.to_diff_dict()

# Copy state_dict (weights + buffers)
copy_to_student(teacher_model=bart_model,
                student_model=distilbart_model,
                config=distilbart_config)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
inputs["output_hidden_states"] = True
# Generate Summary
outputs = bart_model(**inputs)
print(len(outputs.decoder_hidden_states), len(outputs.encoder_hidden_states))

outputs = distilbart_model(**inputs)
print(len(outputs.decoder_hidden_states), len(outputs.encoder_hidden_states))

"""
{'loss': 13.0412, 'learning_rate': 4e-05, 'epoch': 1.0}
{'eval_loss': 9.94822883605957, 'eval_rouge1': 5.8824, 'eval_rouge2': 0.0, 'eval_rougeL': 5.8824, 'eval_rougeLsum': 5.8824, 'eval_gen_len': 62.0, 'eval_runtime': 6.2766, 'eval_samples_per_second': 0.159, 'epoch': 1.0}
{'loss': 8.5078, 'learning_rate': 3e-05, 'epoch': 2.0}
{'eval_loss': 8.609950065612793, 'eval_rouge1': 5.5556, 'eval_rouge2': 0.0, 'eval_rougeL': 5.5556, 'eval_rougeLsum': 5.5556, 'eval_gen_len': 62.0, 'eval_runtime': 6.2624, 'eval_samples_per_second': 0.16, 'epoch': 2.0}
{'loss': 6.7517, 'learning_rate': 2e-05, 'epoch': 3.0}
{'eval_loss': 7.975442409515381, 'eval_rouge1': 0.0, 'eval_rouge2': 0.0, 'eval_rougeL': 0.0, 'eval_rougeLsum': 0.0, 'eval_gen_len': 62.0, 'eval_runtime': 6.2724, 'eval_samples_per_second': 0.159, 'epoch': 3.0}
{'loss': 5.4498, 'learning_rate': 1e-05, 'epoch': 4.0}
{'eval_loss': 7.734518527984619, 'eval_rouge1': 0.0, 'eval_rouge2': 0.0, 'eval_rougeL': 0.0, 'eval_rougeLsum': 0.0, 'eval_gen_len': 62.0, 'eval_runtime': 6.5054, 'eval_samples_per_second': 0.154, 'epoch': 4.0}
{'loss': 4.8699, 'learning_rate': 0.0, 'epoch': 5.0}
{'eval_loss': 7.605877876281738, 'eval_rouge1': 2.9412, 'eval_rouge2': 0.0, 'eval_rougeL': 2.9412, 'eval_rougeLsum': 2.9412, 'eval_gen_len': 62.0, 'eval_runtime': 8.9277, 'eval_samples_per_second': 0.112, 'epoch': 5.0}
{'train_runtime': 71.1609, 'train_samples_per_second': 0.07, 'epoch': 5.0}
"""

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForQuestionAnswering
from autoregkd.models.distilbart.configuration_distilbart import DistilBartConfig
from autoregkd.models.distilbart.modeling_distilbart import *
import torch
import torch.nn as nn


"""
# BART model
bart_model = BartForQuestionAnswering.from_pretrained('facebook/bart-large-xsum')
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


question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='pt')
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])

outputs = distilbart_model(**inputs, start_positions=start_positions, end_positions=end_positions, output_hidden_states=True)
print(len(outputs.decoder_hidden_states))
print(len(outputs.encoder_hidden_states))
print(outputs.decoder_hidden_states[0] == outputs.encoder_hidden_states[0])
print(inputs["input_ids"], inputs["attention_mask"])


import torch
from autoregkd.models.interpolation.modeling_interpolation import LinearInterpolationModule


a = torch.ones(2, 3)
b = 2 * torch.ones(2, 3)
print(a)
print(b)
linear = LinearInterpolationModule(0.25, None)
print(linear(a, b))

bart_model = BartForQuestionAnswering.from_pretrained('Primer/bart-squad2')
for p in bart_model.model.decoder.parameters():
    print(p.requires_grad)
    break
print(len(bart_model.model.decoder.layers))

from autoregkd.models.interpolation.modeling_interpolation import LinearInterpolationModule, InterpolationScheduler

class WrapperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.interp = LinearInterpolationModule(p=0.5)

    def forward(self, a, b):
        return self.interp(a, b)


a = torch.ones(2, 3)
b = 2 * torch.ones(2, 3)
model = WrapperModule()
model.train()
print(model(a, b))
model.eval()
print(model(a, b))
model.train()
print(model(a, b))

sch = InterpolationScheduler(
    interpolation_modules=[LinearInterpolationModule(0.) for _ in range(3)],
    num_interpolation_steps=30,
    max_prob=1,
    per_level_annealing_duration=0.3,
    step_size=5
)
print(sch.per_level_annealing_steps)
print(sch.slopes)
print(sch.starting_points)

for i in range(30):
    print("Step", i)
    for j, m in enumerate(sch.modules):
        print("Module {}: {}".format(j, m.p.data))
    sch.step()
    
"""
print(torch.tensor(0.5))
print(torch.normal(mean=0, std=0.5, size=()).item())


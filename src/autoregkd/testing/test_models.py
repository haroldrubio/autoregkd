from json import decoder
import unittest
import torch
import numpy as np

from transformers import (
    BartModel
)

from src.autoregkd.models.custom_bart import(
    InterpolationModule,
    DistilBartConfig,
    DistilBartDecoder,
    InterpolationDecoder,
    DistilModelOutputWithPastAndCrossAttentions
)

class TestInterpolation(unittest.TestCase):
    def setUp(self):
        self.interp = InterpolationModule()

    def test_no_change_trivial(self):
        sum_of_units = 9
        path_1 = torch.ones(3, 3)
        path_2 = torch.zeros(3, 3)
        swap_1, swap_2 = self.interp(path_1, path_2)
        sum_of_swap = torch.sum(swap_1) + torch.sum(swap_2)
        self.assertEqual(sum_of_units, sum_of_swap)

    def test_same_shape(self):
        common_shape = (12, 15)
        path_1 = torch.rand(common_shape)
        path_2 = torch.rand(common_shape)
        swap_1, swap_2 = self.interp(path_1, path_2)
        self.assertEqual(swap_1.shape, common_shape)
        self.assertEqual(swap_2.shape, common_shape)

    def test_half_mixing(self):
        common_shape = (100, 100)
        total_elements = common_shape[0] * common_shape[1]
        swap_probability = 0.5
        path_1 = torch.ones(common_shape)
        path_2 = torch.zeros(common_shape)
        swap_1, swap_2 = self.interp(path_1, path_2, swap_probability)
        self.assertAlmostEqual(int(torch.sum(swap_1).detach().cpu())/total_elements, total_elements*swap_probability/total_elements, places=2)
        self.assertAlmostEqual(int(torch.sum(swap_2).detach().cpu())/total_elements, total_elements*swap_probability/total_elements, places=2)

    def test_many_mixing(self):
        common_shape = (100, 100)
        total_elements = common_shape[0] * common_shape[1]
        for swap_probability in np.linspace(0, 1, 11):
            path_1 = torch.ones(common_shape)
            path_2 = torch.zeros(common_shape)
            swap_1, swap_2 = self.interp(path_1, path_2, swap_probability)
            self.assertAlmostEqual(int(torch.sum(swap_1).detach().cpu())/total_elements, total_elements*(1-swap_probability)/total_elements, places=1)
            self.assertAlmostEqual(int(torch.sum(swap_2).detach().cpu())/total_elements, total_elements*swap_probability/total_elements, places=1)
        
class TestDecoder(unittest.TestCase):
    def setUp(self):
        model_name = 'facebook/bart-large'
        encoder_layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        decoder_layer_indices = [0, 6, 11]
        config = DistilBartConfig().from_pretrained(model_name)
        config.set_distillation(encoder_layer_indices, decoder_layer_indices)
        bart_model = BartModel.from_pretrained(model_name)
        self.distil_decoder = DistilBartDecoder(config, bart_model.decoder, bart_model.shared)
        self.inter_decoder = InterpolationDecoder(config, bart_model.decoder, bart_model.shared)

        batch_size = 16
        seq_len = 384
        d_hid = 512
        self.decoder_input_ids = torch.randint(0, 10, (batch_size, seq_len))
        self.decoder_attention_mask = torch.ones((batch_size, seq_len))
        self.encoder_hidden_state = torch.rand((batch_size, seq_len, d_hid))
        self.encoder_attention_mask = torch.ones((batch_size, seq_len))
        self.output_attentions = True
        self.output_hidden_states = True

    def test_forward(self):
        output = self.inter_decoder(
            input_ids=self.decoder_input_ids,
            attention_mask=self.decoder_attention_mask,
            encoder_hidden_states=self.encoder_hidden_state,
            encoder_attention_mask=self.encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        print(output.shape)

def run_tests():
    print('running_tests')
    unittest.main()
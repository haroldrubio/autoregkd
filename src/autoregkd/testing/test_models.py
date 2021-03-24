from json import decoder
import unittest
import torch
import numpy as np

from transformers import (
    BartModel
)

from src.autoregkd.models.custom_bart import(
    InterpolationModule,
    InterpolationScheduler,
    DistilBartConfig,
    DistilBartDecoder,
    InterpolationDecoder,
    DistilModelOutputWithPastAndCrossAttentions
)

from src.autoregkd.utils.distil_utils import(
    create_qa_student_by_copying_alternating_layers,
    freeze_params
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
        
class TestIdenticalDecoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'a-ware/bart-squadv2'
        # Create a distilled decoder
        model, _, _ = create_qa_student_by_copying_alternating_layers(
            teacher=model_name,
            d=3,
        )
        self.distil_decoder = model.model.decoder
        del model
        # Create an interpolated decoder
        model, _, _ = create_qa_student_by_copying_alternating_layers(
            teacher=model_name,
            d=3,
            dec_interpolate=True
        )
        self.inter_decoder = model.model.decoder
        del model

        self.distil_decoder.to(self.device)
        self.inter_decoder.to(self.device)

        freeze_params(self.distil_decoder)
        freeze_params(self.inter_decoder)

        batch_size = 16
        seq_len = 384
        d_hid = 1024
        self.decoder_input_ids = torch.randint(0, 10, (batch_size, seq_len), device=self.device)
        self.encoder_hidden_state = torch.rand((batch_size, seq_len, d_hid), device=self.device)
        self.encoder_attention_mask = torch.ones((batch_size, seq_len), device=self.device)
        self.output_attentions = True
        self.output_hidden_states = True

        self.distil = self.distil_decoder(
            input_ids=self.decoder_input_ids,
            encoder_hidden_states=self.encoder_hidden_state,
            encoder_attention_mask=self.encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        self.inter = self.inter_decoder(
            input_ids=self.decoder_input_ids,
            encoder_hidden_states=self.encoder_hidden_state,
            encoder_attention_mask=self.encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    def test_num_states(self):
        self.assertEqual(len(self.inter.hidden_states), len(self.distil.hidden_states))
        self.assertEqual(len(self.inter.cross_attentions), len(self.distil.cross_attentions))
    
    def test_shapes(self):
        self.assertEqual(tuple(self.inter.last_hidden_state.shape),
                         tuple(self.distil.last_hidden_state.shape))
        for hi, hd in zip(self.inter.hidden_states, self.distil.hidden_states):
            si = tuple(hi.shape)
            sd = tuple(hd.shape)
            self.assertEqual(si, sd)
        for ai, ad in zip(self.inter.cross_attentions, self.distil.cross_attentions):
            si = tuple(ai.shape)
            sd = tuple(ad.shape)
            self.assertEqual(si, sd)

    def test_distinct_norms(self):
        self.assertNotEqual(torch.norm(self.inter.last_hidden_state),
                            torch.norm(self.inter.teacher_hidden_state))

    def test_norms(self):
        self.assertAlmostEqual(float(torch.norm(self.inter.last_hidden_state)),
                         float(torch.norm(self.distil.last_hidden_state)), places=-2)
        idx = 0
        for hi, hd in zip(self.inter.hidden_states, self.distil.hidden_states):
            self.assertAlmostEqual(float(torch.norm(hi)),
                             float(torch.norm(hd)), msg=f'not equal hidden at {idx}',places=-2)
            idx += 1
        idx = 0
        for ai, ad in zip(self.inter.cross_attentions, self.distil.cross_attentions):
            self.assertAlmostEqual(float(torch.norm(ai)),
                             float(torch.norm(ad)),  msg=f'not equal attn at {idx}', places=-2)
            idx += 1
    def test_print_norms(self):
        print(f'inter_std {torch.norm(self.inter.last_hidden_state)}')
        print(f'inter_tch {torch.norm(self.inter.teacher_hidden_state)}')
        print(f'distil {torch.norm(self.distil.last_hidden_state)}')

class TestDifferentDecoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'a-ware/bart-squadv2'
        # Create a distilled decoder
        model, _, _ = create_qa_student_by_copying_alternating_layers(
            teacher=model_name,
            d=3,
        )
        self.distil_decoder = model.model.decoder
        del model
        # Create an interpolated decoder
        model, _, _ = create_qa_student_by_copying_alternating_layers(
            teacher=model_name,
            d=3,
            dec_interpolate=True,
            swap_prob=1
        )
        self.inter_decoder = model.model.decoder
        del model

        self.distil_decoder.to(self.device)
        self.inter_decoder.to(self.device)

        freeze_params(self.distil_decoder)
        freeze_params(self.inter_decoder)

        batch_size = 16
        seq_len = 384
        d_hid = 1024
        self.decoder_input_ids = torch.randint(0, 10, (batch_size, seq_len), device=self.device)
        self.encoder_hidden_state = torch.rand((batch_size, seq_len, d_hid), device=self.device)
        self.encoder_attention_mask = torch.ones((batch_size, seq_len), device=self.device)
        self.output_attentions = True
        self.output_hidden_states = True

        self.distil = self.distil_decoder(
            input_ids=self.decoder_input_ids,
            encoder_hidden_states=self.encoder_hidden_state,
            encoder_attention_mask=self.encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )
        self.inter = self.inter_decoder(
            input_ids=self.decoder_input_ids,
            encoder_hidden_states=self.encoder_hidden_state,
            encoder_attention_mask=self.encoder_attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    def test_num_states(self):
        self.assertEqual(len(self.inter.hidden_states), len(self.distil.hidden_states))
        self.assertEqual(len(self.inter.cross_attentions), len(self.distil.cross_attentions))
    
    def test_shapes(self):
        self.assertEqual(tuple(self.inter.last_hidden_state.shape),
                         tuple(self.distil.last_hidden_state.shape))
        for hi, hd in zip(self.inter.hidden_states, self.distil.hidden_states):
            si = tuple(hi.shape)
            sd = tuple(hd.shape)
            self.assertEqual(si, sd)
        for ai, ad in zip(self.inter.cross_attentions, self.distil.cross_attentions):
            si = tuple(ai.shape)
            sd = tuple(ad.shape)
            self.assertEqual(si, sd)

    def test_teacher_norms(self):
        self.assertNotEqual(torch.norm(self.inter.last_hidden_state),
                            torch.norm(self.inter.teacher_hidden_state))

    def test_distinct_norms(self):
        self.assertNotAlmostEqual(float(torch.norm(self.inter.last_hidden_state)),
                         float(torch.norm(self.distil.last_hidden_state)), places=-2)
        for hi, hd in zip(self.inter.hidden_states, self.distil.hidden_states):
            self.assertNotAlmostEqual(float(torch.norm(hi)),
                             float(torch.norm(hd)), places=-2)
        for ai, ad in zip(self.inter.cross_attentions, self.distil.cross_attentions):
            self.assertNotAlmostEqual(float(torch.norm(ai)),
                             float(torch.norm(ad)),places=-2)
    
    def test_print_norms(self):
        print(f'inter_std {torch.norm(self.inter.last_hidden_state)}')
        print(f'inter_tch {torch.norm(self.inter.teacher_hidden_state)}')
        print(f'distil {torch.norm(self.distil.last_hidden_state)}')

class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = 'a-ware/bart-squadv2'
        # Create an interpolated decoder
        model, _, _ = create_qa_student_by_copying_alternating_layers(
            teacher=model_name,
            d=3,
            dec_interpolate=True,
            swap_prob=0.0
        )
        self.inter_decoder = model.model.decoder
        mod_1 = {'start_wu': 0, 'end_wu': 5, 'start_cd': 20, 'end_cd': 30, 'max_prob': 0.5}
        mod_2 = {'start_wu': 5, 'end_wu': 10, 'start_cd': 20, 'end_cd': 30, 'max_prob': 0.25}
        sch_dict = {0: mod_1, 1: mod_2}
        self.scheduler = InterpolationScheduler(
            self.inter_decoder.interp,
            sch_dict,
            40
        )

        self.inter_decoder.to(self.device)

    def test_different_val(self):
        for mod in self.inter_decoder.interp:
            print(mod.swap_prob)
        self.scheduler.step()
        for mod in self.inter_decoder.interp:
            print(mod.swap_prob)
    
    def test_run_scheduler(self):
        for i in range(40):
            out_str = f'{i + 1}| '
            for mod in self.inter_decoder.interp:
                out_str += f'{mod.swap_prob:.3f}, '
            print(out_str)
            self.scheduler.step()

def run_tests():
    print('running_tests')
    unittest.main()
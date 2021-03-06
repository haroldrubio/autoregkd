import unittest
import torch
import numpy as np

from src.autoregkd.models.custom_bart import InterpolationModule

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
        
def run_tests():
    print('running_tests')
    unittest.main()
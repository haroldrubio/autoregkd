import unittest
import random

from src.autoregkd.utils.dataset_utils import *
from src.autoregkd.utils.dataset import SquadDataset

class SquadTestCase(unittest.TestCase):
    def test_load_squad(self):
        num_tests = 100

        tok = BartTokenizerFast.from_pretrained('facebook/bart-base')
        s = SquadDataset('data/squad/sq_dev.json')
        _, q, a = hf_read_squad('data/squad/sq_dev.json')
        num_examples = len(s)
        for t in range(num_tests):
            ex_num = random.randint(0, num_examples - 1)
            text_tens, start, end = s[ex_num]['input_ids'], s[ex_num]['start_positions'], s[ex_num]['end_positions']
            text = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(text_tens)[int(start):int(end)]).strip()
            gt_raw_text = a[ex_num]['text'].strip()
            self.assertEqual(text, gt_raw_text)

def run_tests():
    unittest.main()
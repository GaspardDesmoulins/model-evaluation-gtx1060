import unittest
from unittest import mock

import torch

from coda_remask import remask_tokens


class RemaskTests(unittest.TestCase):
    def test_remasks_only_generated_tokens_with_decaying_probability(self):
        tokens = torch.tensor([[10, 11, 12, 3]])
        draws = torch.tensor([[0.1, 0.2, 0.9]])
        with mock.patch("coda_remask.torch.rand", return_value=draws):
            result = remask_tokens(tokens, 1, 3, 1, 4, 1.0)
        self.assertIs(result, tokens)
        self.assertEqual(tokens.tolist(), [[10, 3, 12, 3]])

    def test_disabled_initial_and_last_step_do_not_remask(self):
        for step, steps, probability in ((0, 3, 0), (None, 3, 1), (2, 3, 1), (0, 1, 1)):
            with self.subTest(step=step, steps=steps, probability=probability):
                tokens = torch.tensor([[10, 11]])
                with mock.patch("coda_remask.torch.rand") as rand:
                    remask_tokens(tokens, 1, 3, step, steps, probability)
                self.assertEqual(tokens.tolist(), [[10, 11]])
                rand.assert_not_called()


if __name__ == "__main__":
    unittest.main()

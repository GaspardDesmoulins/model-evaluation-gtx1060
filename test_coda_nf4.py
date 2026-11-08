import types
import unittest

import bitsandbytes as bnb
import torch

from coda_nf4 import NonFiniteLogitsError, configure_nf4, stable_logits


class NF4Tests(unittest.TestCase):
    def test_activations_use_fp32_and_head_uses_weight_dtype(self):
        embedding = torch.nn.Embedding(2, 2).half()
        head = torch.nn.Linear(2, 2).half()
        quantized = bnb.nn.Linear4bit(2, 2, compute_dtype=torch.float16)
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=embedding),
            lm_head=head,
            modules=lambda: [quantized],
        )
        configure_nf4(model)
        hidden = embedding(torch.tensor([[0]]))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual(head(hidden).dtype, torch.float16)
        self.assertEqual(embedding.weight.dtype, torch.float16)
        self.assertEqual(head.weight.dtype, torch.float16)
        self.assertEqual(quantized.compute_dtype, torch.float32)
        self.assertTrue(quantized.compute_type_is_set)

    def test_logits_are_stable_for_small_temperature(self):
        logits = torch.tensor([[[100.0, -100.0]]], dtype=torch.float16)
        result = stable_logits(torch.tensor([[3]]), logits, 3)
        self.assertEqual(result.dtype, torch.float32)
        probabilities = torch.softmax(result / 1e-6, dim=-1)
        self.assertTrue(torch.isfinite(probabilities).all().item())
        self.assertEqual(probabilities.argmax(dim=-1).item(), 0)

    def test_only_masked_nonfinite_logits_are_rejected(self):
        logits = torch.tensor([[[float("nan"), 1.0], [1.0, 0.0]]])
        tokens = torch.tensor([[7, 3]])
        stable_logits(tokens, logits, 3)
        with self.assertRaises(NonFiniteLogitsError):
            stable_logits(torch.tensor([[3, 3]]), logits, 3)


if __name__ == "__main__":
    unittest.main()

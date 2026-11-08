import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import coda_fp32_hybrid


def sample_tokens(logits, **kwargs):
    return logits.max(dim=-1)


class Layer(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 1


class Rotary(torch.nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, hidden_states


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 3)
        self.model.rotary_emb = Rotary()
        self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        self.model.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(3, 4)

    def _sample(self):
        pass


class HybridTests(unittest.TestCase):
    def test_forward_and_diffusion_preserve_sequence_on_each_boundary(self):
        real_device = torch.device
        torch_proxy = mock.Mock(wraps=torch)
        torch_proxy.device.side_effect = lambda name: real_device(
            "cpu" if name == "cuda" else name
        )
        with mock.patch.object(coda_fp32_hybrid, "torch", torch_proxy):
            for gpu_layers in (0, 1, 2):
                with self.subTest(gpu_layers=gpu_layers):
                    model = Model()
                    coda_fp32_hybrid.place_hybrid(model, gpu_layers)
                    tokens = torch.tensor([[0, 1]])
                    logits, _ = model(tokens)
                    self.assertEqual(tuple(logits.shape), (1, 2, 4))

                    config = types.SimpleNamespace(
                        max_length=4, mask_token_id=3, steps=2, eps=0.001,
                        alg="origin", temperature=0, top_p=1, top_k=0,
                    )
                    generated = model._sample(
                        tokens, None, config,
                        lambda step, x, logits: x,
                        lambda step, x, logits: logits,
                    )
                    self.assertEqual(tuple(generated.shape), (1, 4))
                    self.assertTrue(torch.equal(generated[:, :2], tokens))

    def test_rejects_invalid_layer_counts(self):
        for gpu_layers in (-1, 3):
            with self.subTest(gpu_layers=gpu_layers), self.assertRaises(ValueError):
                coda_fp32_hybrid.place_hybrid(Model(), gpu_layers)

    def test_progress_reports_transferred_layers(self):
        torch_proxy = mock.Mock(wraps=torch)
        torch_proxy.device.return_value = torch.device("cpu")
        messages = []
        with mock.patch.object(coda_fp32_hybrid, "torch", torch_proxy):
            coda_fp32_hybrid.place_hybrid(Model(), 2, progress=messages.append)
        self.assertIn("embedding", messages[0])
        self.assertEqual(messages[1:], [
            "CoDa : blocs GPU transferes 1/2",
            "CoDa : blocs GPU transferes 2/2",
        ])


if __name__ == "__main__":
    unittest.main()

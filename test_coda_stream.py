import contextlib
import io
import json
import runpy
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock


class Tensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Tensor(self.ids[key[1]])
        return self.ids

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.ids


class Tokenizer:
    def __call__(self, text, return_tensors):
        encoded = types.SimpleNamespace(input_ids=Tensor([10]))
        encoded.to = lambda device: encoded
        return encoded

    def decode(self, ids, skip_special_tokens):
        return "".join({10: "P", 11: "A", 12: "B"}.get(i, "") for i in ids)


class Model:
    mask_token_id = 3

    def __init__(self):
        self.waiting = threading.Event()
        self.release = threading.Event()
        self.fail = False
        self.failure = None
        self.temperature = None
        self.logits_hook = None

    def diffusion_generate(self, inputs, max_new_tokens, steps, temperature, **kwargs):
        self.temperature = temperature
        self.logits_hook = kwargs.get("generation_logits_hook_func")
        if self.failure:
            raise self.failure
        if self.fail:
            raise RuntimeError("generation failed")
        hook = kwargs.get("generation_tokens_hook_func")
        if hook:
            initial = Tensor([10, 3, 3])
            assert hook(None, initial, None) is initial
            intermediate = Tensor([10, 11, 3])
            assert hook(0, intermediate, None) is intermediate
            self.waiting.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("test did not release generation")
            final = Tensor([10, 11, 12])
            assert hook(1, final, None) is final
        return Tensor([10, 11, 12])


class DummyServer:
    def __init__(self, address, handler):
        pass

    def serve_forever(self):
        pass


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        namespace = self.load_module("nf4")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), namespace["Handler"])
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def load_module(self, precision):
        config = types.SimpleNamespace(auto_map={}, num_hidden_layers=28)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        fake_torch.float32 = object()
        fake_torch.finfo = lambda dtype: types.SimpleNamespace(tiny=1.1754943508222875e-38)
        fake_torch.inference_mode = contextlib.nullcontext
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: config
        )
        fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: self.model
        )
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: Tokenizer()
        )
        fake_hybrid = types.ModuleType("coda_fp32_hybrid")
        fake_hybrid.place_hybrid = lambda *args, **kwargs: None
        fake_nf4 = types.ModuleType("coda_nf4")
        fake_nf4.NonFiniteLogitsError = type("NonFiniteLogitsError", (RuntimeError,), {})
        self.nonfinite_error = fake_nf4.NonFiniteLogitsError
        fake_nf4.stable_logits = lambda tokens, logits, mask_token_id: logits
        fake_nf4.configure_nf4 = lambda model: setattr(model, "fp32_activations", True)
        modules = {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "coda_fp32_hybrid": fake_hybrid,
            "coda_nf4": fake_nf4,
        }
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch("http.server.ThreadingHTTPServer", DummyServer),
            mock.patch.object(sys, "argv", ["serve_coda.py", "--precision", precision]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            namespace = runpy.run_path(str(Path(__file__).with_name("serve_coda.py")))
        return namespace

    def tearDown(self):
        self.model.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, **params):
        payload = {"messages": [{"role": "user", "content": "P"}], **params}
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=5)

    @staticmethod
    def event(response):
        event_type = response.readline().decode().strip().removeprefix("event: ")
        data = response.readline().decode().strip().removeprefix("data: ")
        assert response.readline() == b"\n"
        return event_type, json.loads(data)

    def test_intermediate_states_arrive_before_generation_finishes(self):
        with self.request("/api/diffusion", diffusion_steps=2, temperature=0.7) as response:
            self.assertEqual(self.event(response), ("start", {"steps": 2}))
            self.assertEqual(
                self.event(response),
                ("progress", {"step": 0, "steps": 2, "text": "[MASK][MASK]"}),
            )
            self.assertEqual(
                self.event(response),
                ("progress", {"step": 1, "steps": 2, "text": "A[MASK]"}),
            )
            self.assertTrue(self.model.waiting.wait(timeout=2))
            self.assertEqual(self.model.temperature, 0.7)
            self.assertIsNotNone(self.model.logits_hook)
            self.model.release.set()
            self.assertEqual(
                self.event(response),
                ("progress", {"step": 2, "steps": 2, "text": "AB"}),
            )
            event, result = self.event(response)
            self.assertEqual(event, "done")
            self.assertEqual(result["text"], "AB")

    def test_standard_api_and_invalid_requests(self):
        with self.request("/v1/chat/completions") as response:
            self.assertEqual(
                json.load(response)["choices"][0]["message"]["content"], "AB"
            )
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/diffusion", diffusion_steps=0)
        self.assertEqual(error.exception.code, 400)

    def test_generation_failure_is_sent_as_an_event(self):
        self.model.fail = True
        with self.request("/api/diffusion", diffusion_steps=2) as response:
            self.assertEqual(self.event(response), ("start", {"steps": 2}))
            self.assertEqual(self.event(response), ("error", {"error": "generation failed"}))

    def test_nf4_temperature_and_logits_hook(self):
        self.assertTrue(self.model.fp32_activations)
        with self.request("/v1/chat/completions", temperature=0.7) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.model.temperature, 0.7)
        self.assertIsNotNone(self.model.logits_hook)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/v1/chat/completions", temperature=1e-40)
        self.assertEqual(error.exception.code, 400)

    def test_nonfinite_logits_are_reported_by_both_routes(self):
        self.model.failure = self.nonfinite_error("non-finite logits")
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/v1/chat/completions", temperature=0.7)
        self.assertEqual(error.exception.code, 500)
        self.assertEqual(json.load(error.exception), {"error": "non-finite logits"})
        with self.request("/api/diffusion", temperature=0.7) as response:
            self.assertEqual(self.event(response)[0], "start")
            self.assertEqual(self.event(response), ("error", {"error": "non-finite logits"}))

    def test_fp32_does_not_use_nf4_hooks(self):
        self.model = Model()
        namespace = self.load_module("fp32")
        namespace["generate"]("P", 2, 2, 0.7)
        self.assertFalse(hasattr(self.model, "fp32_activations"))
        self.assertIsNone(self.model.logits_hook)
        self.assertEqual(self.model.temperature, 0.7)


if __name__ == "__main__":
    unittest.main()

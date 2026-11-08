#!/usr/bin/env python3
"""Serve CoDa NF4 or hybrid FP32 through a local chat endpoint."""

import argparse
import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

print("CoDa : initialisation du runtime Python et CUDA...", flush=True)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from coda_fp32_hybrid import place_hybrid

DEFAULT_NF4_MODEL = Path("/mnt/d/dory/models/CoDA-v0-Instruct-NF4")
DEFAULT_FP32_MODEL = Path("/mnt/d/dory/models/CoDA-v0-Instruct")
CHAT_PAGE = Path(__file__).with_name("coda_chat.html")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path)
parser.add_argument("--precision", choices=("nf4", "fp32"), default="nf4")
parser.add_argument("--gpu-layers", type=int, default=8)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--temperature", type=float, default=0.0)
args = parser.parse_args()

if args.steps < 1 or args.max_tokens < 1:
    parser.error("--steps and --max-tokens must be positive")
if not math.isfinite(args.temperature) or args.temperature < 0:
    parser.error("--temperature must be finite and non-negative")
if args.precision == "nf4" and args.temperature != 0:
    parser.error("NF4 only supports temperature 0 on this runtime")

if args.model is None:
    args.model = DEFAULT_FP32_MODEL if args.precision == "fp32" else DEFAULT_NF4_MODEL

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the CoDa server.")

print(f"CoDa : lecture de la configuration ({args.model})...", flush=True)
config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
config.attention_kernel = "math"
config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
if args.precision == "fp32":
    if not 0 <= args.gpu_layers <= config.num_hidden_layers:
        parser.error(f"--gpu-layers must be between 0 and {config.num_hidden_layers}")
    print("CoDa : preparation des poids FP32 (lecture en RAM a la demande)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    place_hybrid(model, args.gpu_layers, progress=lambda message: print(message, flush=True))
    input_device = "cpu"
else:
    print("CoDa : chargement des poids NF4 sur GPU...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    input_device = "cuda"
print("CoDa : chargement du tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
generation_lock = threading.Lock()


def generate(prompt, max_tokens, steps, temperature, on_step=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    started = time.perf_counter()
    generation_kwargs = {}
    if on_step is not None:
        prompt_length = inputs.input_ids.shape[-1]

        def tokens_hook(step, tokens, logits):
            token_ids = tokens[0, prompt_length:].detach().cpu().tolist()
            parts = []
            unmasked = []
            for token_id in token_ids:
                if token_id == model.mask_token_id:
                    if unmasked:
                        parts.append(tokenizer.decode(unmasked, skip_special_tokens=True))
                        unmasked = []
                    parts.append("[MASK]")
                else:
                    unmasked.append(token_id)
            if unmasked:
                parts.append(tokenizer.decode(unmasked, skip_special_tokens=True))
            on_step(0 if step is None else step + 1, "".join(parts))
            return tokens

        generation_kwargs["generation_tokens_hook_func"] = tokens_hook

    with generation_lock, torch.inference_mode():
        output = model.diffusion_generate(
            inputs=inputs.input_ids,
            max_new_tokens=max_tokens,
            steps=steps,
            temperature=temperature,
            **generation_kwargs,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    completion = text.removeprefix(prompt).strip()
    return completion, output.shape[-1] - inputs.input_ids.shape[-1], time.perf_counter() - started


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = CHAT_PAGE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/api/config":
            self.send_json(
                HTTPStatus.OK,
                {
                    "model": args.model.name,
                    "precision": args.precision,
                    "gpu_layers": args.gpu_layers if args.precision == "fp32" else None,
                    "steps": args.steps,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/api/diffusion"):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            messages = payload["messages"]
            prompt = next(
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            )
            max_tokens = int(payload.get("max_tokens", args.max_tokens))
            steps = int(payload.get("diffusion_steps", args.steps))
            temperature = float(payload.get("temperature", args.temperature))
            if max_tokens < 1 or steps < 1:
                raise ValueError("max_tokens and diffusion_steps must be positive")
            if not math.isfinite(temperature) or temperature < 0:
                raise ValueError("temperature must be finite and non-negative")
            if args.precision == "nf4" and temperature != 0:
                raise ValueError("NF4 only supports temperature 0 on this runtime")
        except (KeyError, StopIteration, TypeError, ValueError, OverflowError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        if self.path == "/api/diffusion":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send_event(event, data):
                body = json.dumps(data, ensure_ascii=True)
                self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))
                self.wfile.flush()

            try:
                send_event("start", {"steps": steps})
                completion, generated_tokens, elapsed = generate(
                    prompt, max_tokens, steps, temperature,
                    on_step=lambda step, text: send_event(
                        "progress", {"step": step, "steps": steps, "text": text}
                    ),
                )
                send_event("done", {
                    "text": completion,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": elapsed,
                })
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except Exception as error:
                try:
                    send_event("error", {"error": str(error)})
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            return

        completion, generated_tokens, elapsed = generate(
            prompt, max_tokens, steps, temperature
        )
        self.send_json(
            HTTPStatus.OK,
            {
                "id": f"coda-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model.name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": inputs_length(prompt),
                    "completion_tokens": generated_tokens,
                    "total_tokens": inputs_length(prompt) + generated_tokens,
                },
                "timings": {"generation_seconds": elapsed, "diffusion_steps": steps},
            },
        )


def inputs_length(prompt):
    return len(tokenizer(prompt, return_tensors="pt").input_ids[0])


server = ThreadingHTTPServer((args.host, args.port), Handler)
print(f"CoDa ready at http://{args.host}:{args.port}", flush=True)
server.serve_forever()

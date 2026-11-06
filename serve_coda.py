#!/usr/bin/env python3
"""Serve the persistent CoDa NF4 checkpoint through a local chat endpoint."""

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = Path("/mnt/d/dory/models/CoDA-v0-Instruct-NF4")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--max-tokens", type=int, default=128)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the CoDa server.")

config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
config.attention_kernel = "math"
config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    trust_remote_code=True,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
generation_lock = threading.Lock()


def generate(prompt, max_tokens, steps):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    started = time.perf_counter()
    with generation_lock, torch.inference_mode():
        output = model.diffusion_generate(
            inputs=inputs.input_ids,
            max_new_tokens=max_tokens,
            steps=steps,
            temperature=0.0,
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
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
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
            if max_tokens < 1 or steps < 1:
                raise ValueError("max_tokens and diffusion_steps must be positive")
        except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return

        completion, generated_tokens, elapsed = generate(prompt, max_tokens, steps)
        self.send_json(
            HTTPStatus.OK,
            {
                "id": f"coda-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "CoDA-v0-Instruct-NF4",
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

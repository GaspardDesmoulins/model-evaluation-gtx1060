#!/usr/bin/env python3
import argparse
import json
import resource
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/mnt/d/dory/models/CoDA-v0-Instruct")
PROMPTS = [
    "Write only Python code for a function add(a: int, b: int) -> int that returns the sum of its arguments.",
    "Write only Python code for a function is_palindrome(text: str) -> bool that ignores case and spaces.",
    "Write only Python code for a function fibonacci(n: int) -> int that returns the nth Fibonacci number iteratively.",
    "Write only Python code for a function unique_sorted(values: list[int]) -> list[int] that returns sorted unique values.",
]


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the CoDa benchmark.")

config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
config.attention_kernel = "math"
config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

load_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    config=config,
    trust_remote_code=True,
    quantization_config=quantization,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
load_seconds = time.perf_counter() - load_start
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

responses = []
for prompt in PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            inputs=inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=0.0,
        )
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(
        {
            "prompt": prompt,
            "response": text,
            "generation_seconds": elapsed,
        }
    )

out = ROOT / "results" / "coda-1.7b-4bit"
out.mkdir(parents=True, exist_ok=True)
(out / "responses.json").write_text(json.dumps(responses, indent=2) + "\n")
(out / "summary.json").write_text(
    json.dumps(
        {
            "model": str(args.model),
            "quantization": "4-bit NF4 at load time from official FP32 weights",
            "attention_backend": "PyTorch math SDPA",
            "diffusion_steps": args.steps,
            "max_new_tokens": args.max_new_tokens,
            "sample_size": len(PROMPTS),
            "load_seconds": load_seconds,
            "mean_generation_seconds": sum(
                item["generation_seconds"] for item in responses
            )
            / len(responses),
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        },
        indent=2,
    )
    + "\n"
)

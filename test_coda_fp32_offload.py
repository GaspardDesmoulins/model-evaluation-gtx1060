#!/usr/bin/env python3
"""Safely test CoDa FP32 with explicit GPU, CPU, and disk placement."""

import json
import resource
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SOURCE = Path("/mnt/d/dory/models/CoDA-v0-Instruct")
OFFLOAD = Path("/mnt/d/dory/models/CoDA-FP32-offload")
RESULT = Path("/home/gaspard/dory/results/coda-fp32-offload")


def load_config():
    config = AutoConfig.from_pretrained(SOURCE, trust_remote_code=True)
    config.attention_kernel = "math"
    config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
    return config


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the FP32 offload test.")

device_map = {
    "model.embed_tokens": 0,
    "model.rotary_emb": 0,
    **{f"model.layers.{index}": 0 for index in range(6)},
    **{f"model.layers.{index}": "cpu" for index in range(6, 23)},
    **{f"model.layers.{index}": "disk" for index in range(23, 28)},
    "model.norm": "cpu",
    "lm_head": "cpu",
}

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(load_config(), trust_remote_code=True)

OFFLOAD.mkdir(parents=True, exist_ok=True)
load_start = time.perf_counter()
model = load_checkpoint_and_dispatch(
    model,
    SOURCE,
    device_map=device_map,
    offload_folder=OFFLOAD,
    offload_state_dict=True,
    dtype=torch.float32,
)
load_seconds = time.perf_counter() - load_start
tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)
prompt = "Write only Python code for a function add(a: int, b: int) -> int."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

forward_start = time.perf_counter()
with torch.inference_mode():
    logits, _ = model(input_ids=inputs.input_ids)
forward_seconds = time.perf_counter() - forward_start

RESULT.mkdir(parents=True, exist_ok=True)
summary = {
    "source": str(SOURCE),
    "dtype": "float32",
    "gpu_layers": 6,
    "cpu_layers": 17,
    "disk_layers": 5,
    "load_seconds": load_seconds,
    "forward_seconds": forward_seconds,
    "logits_shape": list(logits.shape),
    "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
    "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}
(RESULT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

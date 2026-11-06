#!/usr/bin/env python3
"""Export and validate a persistent 4-bit NF4 CoDa checkpoint."""

import json
import shutil
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


SOURCE = Path("/mnt/d/dory/models/CoDA-v0-Instruct")
EXPORT = Path("/mnt/d/dory/models/CoDA-v0-Instruct-NF4")
PROMPT = "Write only Python code for a function add(a: int, b: int) -> int."


def load_config(path):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.attention_kernel = "math"
    config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
    return config


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the CoDa NF4 export.")

EXPORT.mkdir(parents=True, exist_ok=True)
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

source_load_start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    SOURCE,
    config=load_config(SOURCE),
    trust_remote_code=True,
    quantization_config=quantization,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
source_load_seconds = time.perf_counter() - source_load_start
tokenizer = AutoTokenizer.from_pretrained(SOURCE, trust_remote_code=True)

export_start = time.perf_counter()
model.generation_config.temperature = None
model.save_pretrained(EXPORT, safe_serialization=True)
tokenizer.save_pretrained(EXPORT)
for source_file in SOURCE.glob("*.py"):
    shutil.copy2(source_file, EXPORT / source_file.name)
export_seconds = time.perf_counter() - export_start

del model
torch.cuda.empty_cache()

reload_start = time.perf_counter()
reloaded = AutoModelForCausalLM.from_pretrained(
    EXPORT,
    config=load_config(EXPORT),
    trust_remote_code=True,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
reload_seconds = time.perf_counter() - reload_start
inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = reloaded.diffusion_generate(
        inputs=inputs.input_ids,
        max_new_tokens=32,
        steps=16,
        temperature=0.0,
    )
response = tokenizer.decode(output[0], skip_special_tokens=True)

summary = {
    "source": str(SOURCE),
    "export": str(EXPORT),
    "quantization": "4-bit NF4",
    "source_load_seconds": source_load_seconds,
    "export_seconds": export_seconds,
    "reload_seconds": reload_seconds,
    "export_bytes": sum(path.stat().st_size for path in EXPORT.rglob("*") if path.is_file()),
    "validation_response": response,
}
(EXPORT / "export-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

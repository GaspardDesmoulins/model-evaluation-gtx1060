#!/usr/bin/env python3
import argparse
import json
import math
import resource
import subprocess
import sys
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
IFEVAL = ROOT / "ifeval-source"
INPUTS = IFEVAL / "instruction_following_eval/data/input_data.jsonl"
DEFAULT_MODEL = Path("/mnt/d/dory/models/CoDA-v0-Instruct")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--limit", type=int, default=16)
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--steps", type=int, default=64)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the CoDa evaluation.")

with INPUTS.open() as file:
    inputs = [json.loads(line) for line in file]
inputs = inputs[::math.ceil(len(inputs) / args.limit)][:args.limit]

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

out = ROOT / "results" / "ifeval-coda-1.7b-4bit"
out.mkdir(parents=True, exist_ok=True)
(out / "inputs.jsonl").write_text("".join(json.dumps(item) + "\n" for item in inputs))

responses = []
for index, item in enumerate(inputs, 1):
    prompt = item["prompt"]
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            inputs=encoded.input_ids,
            max_new_tokens=args.max_tokens,
            steps=args.steps,
            temperature=0.0,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    responses.append(
        {
            "prompt": prompt,
            "response": text.removeprefix(prompt).strip(),
            "generation_seconds": time.perf_counter() - start,
        }
    )
    print(f"CoDa: {index}/{len(inputs)}", flush=True)

with (out / "responses.jsonl").open("w") as file:
    for response in responses:
        file.write(json.dumps(response) + "\n")

subprocess.run(
    [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={out / 'inputs.jsonl'}",
        f"--input_response_data={out / 'responses.jsonl'}",
        f"--output_dir={out}",
    ],
    cwd=IFEVAL,
    check=True,
)

scores = {}
for mode in ("strict", "loose"):
    results = [json.loads(line) for line in (out / f"eval_results_{mode}.jsonl").open()]
    scores[mode] = sum(item["follow_all_instructions"] for item in results) / len(results)

(out / "summary.json").write_text(
    json.dumps(
        {
            "model": str(args.model),
            "quantization": "4-bit NF4 at load time from official FP32 weights",
            "attention_backend": "PyTorch math SDPA",
            "sample_size": len(inputs),
            "max_tokens": args.max_tokens,
            "diffusion_steps": args.steps,
            "load_seconds": load_seconds,
            "mean_generation_seconds": sum(
                item["generation_seconds"] for item in responses
            )
            / len(responses),
            "strict_prompt_accuracy": scores["strict"],
            "loose_prompt_accuracy": scores["loose"],
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        },
        indent=2,
    )
    + "\n"
)

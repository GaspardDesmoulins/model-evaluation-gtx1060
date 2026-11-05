#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IFEVAL = ROOT / "ifeval-source"
INPUTS = IFEVAL / "instruction_following_eval/data/input_data.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--limit", type=int, default=16)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_args()

with INPUTS.open() as file:
    inputs = [json.loads(line) for line in file]
inputs = inputs[::math.ceil(len(inputs) / args.limit)][:args.limit]
name = args.model.replace(":", "-")
out = ROOT / "results" / f"ifeval-{name}"
out.mkdir(parents=True, exist_ok=True)
(out / "inputs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inputs))

def chat(prompt):
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({"model": args.model, "messages": [{"role": "user", "content": prompt}],
                         "stream": False, "options": {"temperature": 0, "num_predict": args.max_tokens}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)["message"]["content"]

peak_ram = peak_vram = 0
with (out / "responses.jsonl").open("w") as file:
    for index, item in enumerate(inputs, 1):
        response = chat(item["prompt"])
        file.write(json.dumps({"prompt": item["prompt"], "response": response}) + "\n")
        file.flush()
        peak_vram = max(peak_vram, int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()))
        print(f"{args.model}: {index}/{len(inputs)}", flush=True)

subprocess.run([sys.executable, "-m", "instruction_following_eval.evaluation_main",
    f"--input_data={out / 'inputs.jsonl'}", f"--input_response_data={out / 'responses.jsonl'}",
    f"--output_dir={out}"], cwd=IFEVAL, check=True)
scores = {}
for mode in ("strict", "loose"):
    results = [json.loads(line) for line in (out / f"eval_results_{mode}.jsonl").open()]
    scores[mode] = sum(x["follow_all_instructions"] for x in results) / len(results)
(out / "summary.json").write_text(json.dumps({"model": args.model, "sample_size": len(inputs),
    "strict_prompt_accuracy": scores["strict"], "loose_prompt_accuracy": scores["loose"],
    "peak_vram_mib": peak_vram}, indent=2) + "\n")

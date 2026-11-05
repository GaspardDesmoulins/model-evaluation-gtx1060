#!/usr/bin/env python3
"""Run a reproducible IFEval sample against a Dory llama.cpp server."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "llama.cpp-dory/build-cpu/bin/llama-server"
DEFAULT_MODEL = ROOT / "models/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
IFEVAL_ROOT = ROOT / "ifeval-source"
INPUTS = IFEVAL_ROOT / "instruction_following_eval/data/input_data.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--server", type=Path, default=SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-layers", default="0")
    parser.add_argument("--fit-target", type=int)
    return parser.parse_args()


def read_inputs(limit):
    with INPUTS.open(encoding="utf-8") as file:
        inputs = [json.loads(line) for line in file]
    stride = math.ceil(len(inputs) / limit)
    return inputs[::stride][:limit]


def request_json(url, payload, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(port, process):
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server stopped with exit code {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)
    raise TimeoutError("llama-server did not become ready within 180 seconds")


def read_rss_bytes(pid):
    with open(f"/proc/{pid}/status", encoding="utf-8") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def gpu_memory_mib():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return max(int(value) for value in result.stdout.splitlines())


def monitor_memory(process, cpu_peak, gpu_peak):
    while process.poll() is None:
        cpu_peak[0] = max(cpu_peak[0], read_rss_bytes(process.pid))
        gpu_peak[0] = max(gpu_peak[0], gpu_memory_mib())
        time.sleep(0.25)


def score_responses(output_dir, responses_file):
    command = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={output_dir / 'inputs.jsonl'}",
        f"--input_response_data={responses_file}",
        f"--output_dir={output_dir}",
    ]
    subprocess.run(command, cwd=IFEVAL_ROOT, check=True)
    scores = {}
    for mode in ("strict", "loose"):
        result_file = output_dir / f"eval_results_{mode}.jsonl"
        with result_file.open(encoding="utf-8") as file:
            results = [json.loads(line) for line in file]
        scores[mode] = {
            "prompt_accuracy": sum(item["follow_all_instructions"] for item in results) / len(results),
            "instruction_accuracy": sum(
                sum(item["follow_instruction_list"]) for item in results
            ) / sum(len(item["follow_instruction_list"]) for item in results),
        }
    return scores


def main():
    args = parse_args()
    if not args.server.is_file() or not args.model.is_file() or not INPUTS.is_file():
        raise FileNotFoundError("Dory server, Q5 model, or official IFEval inputs are missing")

    output_dir = ROOT / "results" / f"ifeval-loop-{args.loops}"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = read_inputs(args.limit)
    with (output_dir / "inputs.jsonl").open("w", encoding="utf-8") as file:
        for item in inputs:
            file.write(json.dumps(item) + "\n")

    log_file = (output_dir / "server.log").open("w", encoding="utf-8")
    command = [
        str(args.server),
        "-m",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-ngl",
        args.gpu_layers,
        "-t",
        str(args.threads),
        "-c",
        "4096",
        "--jinja",
        "--reasoning",
        "off",
        "--override-kv",
        f"dory.num_loops=int:{args.loops},dory.recurrent_kv_cache_mode=str:per_loop",
    ]
    if args.fit_target is not None:
        command.extend(["--fit", "on", "--fit-target", str(args.fit_target)])
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    peak_rss = [0]
    peak_vram_mib = [0]
    monitor = threading.Thread(
        target=monitor_memory, args=(process, peak_rss, peak_vram_mib), daemon=True
    )
    monitor.start()

    try:
        wait_for_server(args.port, process)
        responses_file = output_dir / "responses.jsonl"
        with responses_file.open("w", encoding="utf-8") as file:
            for index, item in enumerate(inputs, start=1):
                result = request_json(
                    f"http://127.0.0.1:{args.port}/v1/chat/completions",
                    {
                        "model": "dory",
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": args.max_tokens,
                    },
                )
                response = result["choices"][0]["message"]["content"]
                file.write(json.dumps({"prompt": item["prompt"], "response": response}) + "\n")
                file.flush()
                print(f"loop {args.loops}: completed {index}/{len(inputs)}", flush=True)

        scores = score_responses(output_dir, responses_file)
        summary = {
            "loops": args.loops,
            "sample_size": len(inputs),
            "max_tokens": args.max_tokens,
            "threads": args.threads,
            "context_size": 4096,
            "kv_cache_mode": "per_loop",
            "peak_rss_bytes": peak_rss[0],
            "peak_vram_mib": peak_vram_mib[0],
            "model": str(args.model),
            "gpu_layers": args.gpu_layers,
            "fit_target_mib": args.fit_target,
            "strict_prompt_accuracy": scores["strict"]["prompt_accuracy"],
            "strict_instruction_accuracy": scores["strict"]["instruction_accuracy"],
            "loose_prompt_accuracy": scores["loose"]["prompt_accuracy"],
            "loose_instruction_accuracy": scores["loose"]["instruction_accuracy"],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
        log_file.close()


if __name__ == "__main__":
    main()

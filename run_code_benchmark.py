#!/usr/bin/env python3
"""Benchmark shared code prompts through Dory llama.cpp or Ollama."""

import argparse
import json
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from code_benchmark_prompts import PROMPTS


ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER = ROOT / "llama.cpp-dory/build-cuda/bin/llama-server"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("dory", "ollama"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--loops", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--fit-target", type=int)
    parser.add_argument("--port", type=int, default=8082)
    return parser.parse_args()


def request_json(url, payload, timeout=900):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_memory_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
    )
    return max(int(value) for value in result.stdout.splitlines())


def read_rss_bytes(pid):
    with open(f"/proc/{pid}/status", encoding="utf-8") as file:
        for line in file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def monitor_memory(process, peak_rss, peak_vram):
    while process.poll() is None:
        peak_rss[0] = max(peak_rss[0], read_rss_bytes(process.pid))
        peak_vram[0] = max(peak_vram[0], gpu_memory_mib())
        time.sleep(0.25)


def wait_for_server(port, process):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server stopped with exit code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)
    raise TimeoutError("llama-server did not become ready within 180 seconds")


def run_dory(args, out):
    log_file = (out / "server.log").open("w", encoding="utf-8")
    command = [
        str(args.server),
        "-m",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-ngl",
        args.gpu_layers,
        "-t",
        "6",
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

    load_start = time.perf_counter()
    process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
    peak_rss = [0]
    peak_vram = [0]
    monitor = threading.Thread(
        target=monitor_memory, args=(process, peak_rss, peak_vram), daemon=True
    )
    monitor.start()

    try:
        wait_for_server(args.port, process)
        load_seconds = time.perf_counter() - load_start
        responses = []
        for prompt in PROMPTS:
            started = time.perf_counter()
            result = request_json(
                f"http://127.0.0.1:{args.port}/v1/chat/completions",
                {
                    "model": "dory",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": args.max_tokens,
                },
            )
            responses.append(
                {
                    "prompt": prompt,
                    "response": result["choices"][0]["message"]["content"],
                    "wall_seconds": time.perf_counter() - started,
                    "usage": result.get("usage", {}),
                }
            )
        return responses, {
            "load_seconds": load_seconds,
            "peak_rss_mib": peak_rss[0] / 1024**2,
            "peak_vram_mib": peak_vram[0],
        }
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=30)
        log_file.close()


def run_ollama(args):
    responses = []
    peak_vram = 0
    for prompt in PROMPTS:
        started = time.perf_counter()
        result = request_json(
            "http://127.0.0.1:11434/api/chat",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": args.max_tokens},
            },
        )
        peak_vram = max(peak_vram, gpu_memory_mib())
        responses.append(
            {
                "prompt": prompt,
                "response": result["message"]["content"],
                "wall_seconds": time.perf_counter() - started,
                "load_seconds": result["load_duration"] / 1e9,
                "prompt_eval_tokens": result["prompt_eval_count"],
                "prompt_eval_seconds": result["prompt_eval_duration"] / 1e9,
                "generated_tokens": result["eval_count"],
                "generation_seconds": result["eval_duration"] / 1e9,
            }
        )
    return responses, {"peak_vram_mib": peak_vram}


def main():
    args = parse_args()
    out = ROOT / "results" / f"code-{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    benchmark_start = time.perf_counter()

    if args.backend == "dory":
        responses, measurements = run_dory(args, out)
    else:
        responses, measurements = run_ollama(args)

    (out / "responses.json").write_text(json.dumps(responses, indent=2) + "\n")
    summary = {
        "backend": args.backend,
        "model": args.model,
        "sample_size": len(PROMPTS),
        "max_tokens": args.max_tokens,
        "total_seconds": time.perf_counter() - benchmark_start,
        "mean_wall_seconds": sum(item["wall_seconds"] for item in responses) / len(responses),
        **measurements,
    }
    if args.backend == "dory":
        summary["loops"] = args.loops
        summary["gpu_layers"] = args.gpu_layers
        summary["fit_target_mib"] = args.fit_target
    else:
        summary["generated_tokens"] = sum(item["generated_tokens"] for item in responses)
        summary["generation_seconds"] = sum(item["generation_seconds"] for item in responses)
        summary["generation_tokens_per_second"] = (
            summary["generated_tokens"] / summary["generation_seconds"]
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

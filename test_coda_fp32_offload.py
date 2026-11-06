#!/usr/bin/env python3
"""Safely test CoDa FP32 with explicit GPU, CPU, and disk placement."""

import json
import resource
import time
import argparse
import types
from pathlib import Path

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SOURCE = Path("/mnt/d/dory/models/CoDA-v0-Instruct")
OFFLOAD = Path("/mnt/d/dory/models/CoDA-FP32-offload")
RESULT_ROOT = Path("/home/gaspard/dory/results")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-layers", type=int, default=6)
parser.add_argument("--disk-layers", type=int, default=5)
parser.add_argument("--diffusion-test", action="store_true")
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument(
    "--manual-gpu-layers",
    type=int,
    help="Run this many initial decoder layers on CUDA without Accelerate hooks.",
)
parser.add_argument("--max-new-tokens", type=int, default=16)
parser.add_argument("--steps", type=int, default=4)
args = parser.parse_args()


def load_config():
    config = AutoConfig.from_pretrained(SOURCE, trust_remote_code=True)
    config.attention_kernel = "math"
    config.auto_map["AutoModelForCausalLM"] = "modeling_coda.CoDALanguageModel"
    return config


if not args.cpu_only and not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the FP32 offload test.")

if (
    not args.cpu_only
    and args.manual_gpu_layers is None
    and (
    not 0 <= args.gpu_layers <= 28
    or not 0 <= args.disk_layers <= 28 - args.gpu_layers
    )
):
    raise ValueError("Invalid GPU or disk layer count")

if args.manual_gpu_layers is not None:
    if not 0 <= args.manual_gpu_layers <= 28:
        raise ValueError("manual-gpu-layers must be between 0 and 28")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for manual GPU placement.")
    device_map = None
    gpu_layers = args.manual_gpu_layers
    cpu_layers = 28 - gpu_layers
    disk_layers = 0
    input_device = "cpu"
elif args.cpu_only:
    device_map = {"": "cpu"}
    gpu_layers = 0
    cpu_layers = 28
    disk_layers = 0
    input_device = "cpu"
else:
    cpu_end = 28 - args.disk_layers
    device_map = {
        "model.embed_tokens": 0,
        "model.rotary_emb": 0,
        **{f"model.layers.{index}": 0 for index in range(args.gpu_layers)},
        **{f"model.layers.{index}": "cpu" for index in range(args.gpu_layers, cpu_end)},
        **{f"model.layers.{index}": "disk" for index in range(cpu_end, 28)},
        "model.norm": "cpu",
        "lm_head": "cpu",
    }
    gpu_layers = args.gpu_layers
    cpu_layers = cpu_end - args.gpu_layers
    disk_layers = args.disk_layers
    input_device = "cuda"

load_start = time.perf_counter()
if args.cpu_only or args.manual_gpu_layers is not None:
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE,
        config=load_config(),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    if args.manual_gpu_layers is not None:
        cuda_device = torch.device("cuda")
        backbone = model.model
        backbone.embed_tokens.to(cuda_device)
        backbone.rotary_emb.to(cuda_device)
        for layer in list(backbone.layers)[:args.manual_gpu_layers]:
            layer.to(cuda_device)

        def hybrid_forward(self, input_ids, **kwargs):
            hidden_states = backbone.embed_tokens(input_ids.to(cuda_device))
            seq_length = hidden_states.size(1)
            position_ids = (
                torch.arange(seq_length, device=cuda_device).unsqueeze(0).float()
            )
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length),
                    float("-inf"),
                    device=cuda_device,
                ),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

            for index, layer in enumerate(backbone.layers):
                if index == args.manual_gpu_layers:
                    hidden_states = hidden_states.to("cpu")
                    causal_mask = causal_mask.to("cpu")
                    position_ids = position_ids.to("cpu")
                    position_embeddings = tuple(
                        embedding.to("cpu") for embedding in position_embeddings
                    )
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

            logits = self.lm_head(backbone.norm(hidden_states))
            return logits, None

        generation_globals = model._sample.__func__.__globals__
        functional = generation_globals["F"]
        sample_tokens = generation_globals["sample_tokens"]

        def hybrid_sample(
            self,
            input_ids,
            attention_mask,
            generation_config,
            generation_tokens_hook_func,
            generation_logits_hook_func,
        ):
            max_length = generation_config.max_length
            mask_token_id = generation_config.mask_token_id
            steps = generation_config.steps
            eps = generation_config.eps
            alg = generation_config.alg
            temperature = generation_config.temperature
            top_p = generation_config.top_p
            top_k = generation_config.top_k
            x = functional.pad(
                input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id
            )
            timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
            x = generation_tokens_hook_func(None, x, None)

            for index in range(steps):
                mask_index = x == mask_token_id
                logits, _ = self(x, attention_mask=attention_mask)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                logits = generation_logits_hook_func(index, x, logits)
                mask_logits = logits[mask_index]
                transfer_probability = (
                    1 - timesteps[index + 1] / timesteps[index]
                    if index < steps - 1
                    else 1
                )

                if alg != "origin":
                    raise RuntimeError(f"Unsupported diffusion algorithm: {alg}")

                x0 = torch.full_like(x[mask_index], mask_token_id)
                transfer_index = torch.rand(
                    x0.shape, device=x.device
                ) < transfer_probability
                _, x0[transfer_index] = sample_tokens(
                    mask_logits[transfer_index],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x[mask_index] = x0
                x = generation_tokens_hook_func(index, x, logits)

            return x

        model.forward = types.MethodType(hybrid_forward, model)
        model._sample = types.MethodType(hybrid_sample, model)
else:
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(load_config(), trust_remote_code=True)
    OFFLOAD.mkdir(parents=True, exist_ok=True)
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
inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

forward_start = time.perf_counter()
with torch.inference_mode():
    logits, _ = model(input_ids=inputs.input_ids)
forward_seconds = time.perf_counter() - forward_start

result = RESULT_ROOT / (
    "coda-fp32-cpu-only"
    if args.cpu_only
    else (
        f"coda-fp32-manual-gpu{args.manual_gpu_layers}"
        if args.manual_gpu_layers is not None
        else f"coda-fp32-gpu{args.gpu_layers}-disk{args.disk_layers}"
    )
)
result.mkdir(parents=True, exist_ok=True)
summary = {
    "source": str(SOURCE),
    "dtype": "float32",
    "gpu_layers": gpu_layers,
    "cpu_layers": cpu_layers,
    "disk_layers": disk_layers,
    "load_seconds": load_seconds,
    "forward_seconds": forward_seconds,
    "logits_shape": list(logits.shape),
    "peak_vram_mib": (
        torch.cuda.max_memory_allocated() / 1024**2
        if not args.cpu_only
        else 0
    ),
    "peak_process_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}
(result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if args.diffusion_test:
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            inputs=inputs.input_ids,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            temperature=0.0,
        )
    summary["diffusion_generation_seconds"] = time.perf_counter() - generation_start
    summary["diffusion_steps"] = args.steps
    summary["max_new_tokens"] = args.max_new_tokens
    summary["response"] = tokenizer.decode(output[0], skip_special_tokens=True)
    summary["peak_vram_mib"] = (
        torch.cuda.max_memory_allocated() / 1024**2
        if not args.cpu_only
        else 0
    )
    summary["peak_process_ram_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

(result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

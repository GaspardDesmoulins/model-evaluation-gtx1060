"""Optional remasking of generated tokens between diffusion steps."""

import math

import torch


def remask_tokens(tokens, prompt_length, mask_token_id, step, steps, probability):
    if probability == 0 or step is None or step >= steps - 1:
        return tokens
    rate = probability * math.exp(-5.0 * step / (steps - 1))
    generated = tokens[:, prompt_length:]
    selected = torch.rand(generated.shape, device=tokens.device) < rate
    generated.masked_fill_(selected, mask_token_id)
    return tokens

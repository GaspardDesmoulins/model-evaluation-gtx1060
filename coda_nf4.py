"""Numerical safeguards for the CoDa NF4 checkpoint."""

import torch


class NonFiniteLogitsError(RuntimeError):
    pass


def configure_nf4(model):
    import bitsandbytes as bnb

    model.model.embed_tokens.register_forward_hook(
        lambda module, inputs, output: output.float()
    )
    model.lm_head.register_forward_pre_hook(
        lambda module, inputs: (inputs[0].to(module.weight.dtype),)
    )
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.compute_dtype = torch.float32
            module.compute_type_is_set = True


def stable_logits(tokens, logits, mask_token_id):
    finite_rows = torch.isfinite(logits).all(dim=-1)
    if not (finite_rows | (tokens != mask_token_id)).all().item():
        raise NonFiniteLogitsError("CoDa NF4 produced non-finite logits")
    logits = logits.float()
    return logits - logits.amax(dim=-1, keepdim=True)

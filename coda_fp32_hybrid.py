"""Manual CoDa FP32 placement for GPU layers followed by CPU layers."""

import types

import torch


def place_hybrid(model, gpu_layers, progress=None):
    backbone = model.model
    if not 0 <= gpu_layers <= len(backbone.layers):
        raise ValueError(f"gpu_layers must be between 0 and {len(backbone.layers)}")

    cuda_device = torch.device("cuda")
    if progress is not None:
        progress("CoDa : transfert de l'embedding sur GPU...")
    backbone.embed_tokens.to(cuda_device)
    backbone.rotary_emb.to(cuda_device)
    for index, layer in enumerate(list(backbone.layers)[:gpu_layers], 1):
        layer.to(cuda_device)
        if progress is not None:
            progress(f"CoDa : blocs GPU transferes {index}/{gpu_layers}")

    def hybrid_forward(self, input_ids, **kwargs):
        hidden_states = backbone.embed_tokens(input_ids.to(cuda_device))
        seq_length = hidden_states.size(1)
        position_ids = torch.arange(seq_length, device=cuda_device).unsqueeze(0).float()
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=cuda_device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        position_embeddings = backbone.rotary_emb(hidden_states, position_ids)

        for index, layer in enumerate(backbone.layers):
            if index == gpu_layers:
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

        logits = self.lm_head(backbone.norm(hidden_states.to("cpu")))
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
            transfer_index = torch.rand(x0.shape, device=x.device) < transfer_probability
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

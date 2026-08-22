import functools
from collections import defaultdict

import einops
import torch
from tqdm import tqdm
from transformer_lens import utils

from argos.activations import tokenize_instructions


def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]


def compute_refusal_directions(harmful_acts, harmless_acts, model, config):
    activation_refusals = defaultdict(list)
    pos = -1

    for layer_num in range(1, model.cfg.n_layers):
        for layer in config.activation_layers:
            harmful_mean = get_act_idx(harmful_acts, layer, layer_num)[:, pos, :].mean(dim=0)
            harmless_mean = get_act_idx(harmless_acts, layer, layer_num)[:, pos, :].mean(dim=0)
            direction = harmful_mean - harmless_mean
            direction = direction / direction.norm()
            activation_refusals[layer].append(direction)

    candidates = [
        activation_refusals[layer][l - 1]
        for l in range(1, model.cfg.n_layers)
        for layer in config.selected_layers
    ]
    return sorted(candidates, key=lambda d: abs(d.mean()), reverse=True)


def direction_ablation_hook(activation, hook, direction):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = (
        einops.einsum(activation, direction.view(-1, 1), "... d_act, d_act single -> ... single")
        * direction
    )
    return activation - proj


def _generate_with_hooks(model, tokenizer, tokens, max_tokens_generated=64, fwd_hooks=()):
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    )
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=list(fwd_hooks)):
            logits = model(all_tokens[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1)
            all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(all_tokens[:, tokens.shape[1]:], skip_special_tokens=True)


def get_generations(model, tokenizer, instructions, fwd_hooks=(), max_tokens_generated=64, batch_size=4):
    generations = []
    for i in range(0, len(instructions), batch_size):
        tokens = tokenize_instructions(tokenizer, instructions[i : i + batch_size])
        generations.extend(
            _generate_with_hooks(model, tokenizer, tokens, max_tokens_generated, fwd_hooks)
        )
    return generations


def score_directions(model, tokenizer, directions, test_instructions, config):
    evals = []
    for direction in tqdm(directions[: config.eval_top_n]):
        hook_fn = functools.partial(direction_ablation_hook, direction=direction)
        fwd_hooks = [
            (utils.get_act_name(act_name, layer), hook_fn)
            for layer in range(model.cfg.n_layers)
            for act_name in config.activation_layers
        ]
        evals.append(get_generations(model, tokenizer, test_instructions, fwd_hooks=fwd_hooks))
    return evals

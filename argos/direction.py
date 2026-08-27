import torch
from tqdm import tqdm

from argos.activations import get_layers, tokenize_instructions

MIN_DIRECTION_NORM = 1e-4


def compute_refusal_directions_indexed(harmful_acts, harmless_acts, config):
    results = []
    for act_name in config.selected_layers:
        for layer_idx in sorted(harmful_acts[act_name]):
            harmful_mean = harmful_acts[act_name][layer_idx].mean(dim=0)
            harmless_mean = harmless_acts[act_name][layer_idx].mean(dim=0)
            direction = harmful_mean - harmless_mean
            norm = direction.norm()
            if norm < MIN_DIRECTION_NORM:
                continue
            direction = direction / norm
            if torch.isnan(direction).any():
                continue
            results.append((act_name, layer_idx, direction))
    return results


def compute_refusal_directions(harmful_acts, harmless_acts, config):
    indexed = compute_refusal_directions_indexed(harmful_acts, harmless_acts, config)
    directions = [direction for _, _, direction in indexed]
    return sorted(directions, key=lambda d: abs(d.mean()).item(), reverse=True)


def direction_ablation(activation, direction):
    direction = direction.to(activation.device, activation.dtype)
    proj = (activation @ direction).unsqueeze(-1) * direction
    return activation - proj


def _ablation_hooks(layers, direction):
    handles = []

    def pre_hook(module, args, kwargs):
        return (direction_ablation(args[0], direction),) + args[1:], kwargs

    def post_hook(module, args, output):
        return direction_ablation(output, direction)

    for layer in layers:
        handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(layer.register_forward_hook(post_hook))
    return handles


def get_generations(model, tokenizer, instructions, direction=None, max_new_tokens=64, batch_size=4):
    device = next(model.parameters()).device
    layers = get_layers(model)
    generations = []

    for i in range(0, len(instructions), batch_size):
        tokens = tokenize_instructions(tokenizer, instructions[i : i + batch_size]).to(device)
        handles = _ablation_hooks(layers, direction) if direction is not None else []
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for h in handles:
            h.remove()
        generations.extend(tokenizer.batch_decode(output[:, tokens.shape[1]:], skip_special_tokens=True))

    return generations


def score_directions(model, tokenizer, directions, test_instructions, config):
    evals = []
    for direction in tqdm(directions[: config.eval_top_n]):
        evals.append(get_generations(model, tokenizer, test_instructions, direction=direction))
    return evals

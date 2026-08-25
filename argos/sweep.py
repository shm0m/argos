import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from argos.ablate import apply_ablation
from argos.activations import collect_activations
from argos.config import ArgosConfig
from argos.data import get_harmful_instructions, get_harmless_instructions
from argos.direction import compute_refusal_directions_indexed, get_generations
from argos.eval.capability import run_capability_benchmarks
from argos.eval.refusal import degenerate_rate, refusal_stats


def pick_evenly_spaced(items, n):
    if n >= len(items):
        return list(items)
    if n <= 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    picked = sorted({round(i * step) for i in range(n)})
    return [items[i] for i in picked]


def sweep(config, n_directions, tasks, limit, output):
    dtype = getattr(torch, config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(config.model_id, torch_dtype=dtype, device_map=config.device)
    model.eval()

    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions(test_size=config.n_test_instructions)
    test_instructions = harmful_test[: config.n_test_instructions]

    harmful_acts, harmless_acts = collect_activations(model, tokenizer, harmful_train, harmless_train, config)
    indexed = compute_refusal_directions_indexed(harmful_acts, harmless_acts, config)
    indexed = sorted(indexed, key=lambda t: t[1])
    candidates = pick_evenly_spaced(indexed, n_directions)

    del model
    torch.cuda.empty_cache()

    results = []
    for act_name, layer_idx, direction in candidates:
        model = AutoModelForImageTextToText.from_pretrained(config.model_id, torch_dtype=dtype, device_map=config.device)
        model.eval()

        apply_ablation(model, direction)

        generations = get_generations(model, tokenizer, test_instructions)
        refusal = refusal_stats(generations)
        degeneracy = degenerate_rate(generations)

        capability = {} if degeneracy > 0.5 else run_capability_benchmarks(model, config.model_id, tasks=tasks, limit=limit)

        entry = {
            "act_name": act_name,
            "layer": layer_idx,
            "refusal": refusal,
            "degenerate_rate": degeneracy,
            "capability": capability,
        }
        results.append(entry)
        print(f"couche {layer_idx}: refus={refusal['rate']:.2%} degenere={degeneracy:.2%}")

        del model
        torch.cuda.empty_cache()

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    parser = argparse.ArgumentParser(prog="argos-sweep")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--n-directions", type=int, default=8)
    parser.add_argument("--tasks", nargs="+", default=["hellaswag"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="results/phase2_sweep.json")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)
    sweep(config, args.n_directions, args.tasks, args.limit, args.output)


if __name__ == "__main__":
    main()

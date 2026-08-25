import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from argos.activations import get_layers, tokenize_instructions
from argos.config import ArgosConfig
from argos.data import get_harmful_instructions


def collect_projection(model, tokenizer, instructions, act_name, layer_idx, direction, batch_size=8):
    device = next(model.parameters()).device
    layer = get_layers(model)[layer_idx]
    direction = direction.to(device, torch.float32)

    projections = []

    def pre_hook(module, args, kwargs):
        act = args[0][:, -1, :].detach().to(torch.float32)
        projections.append((act @ direction).cpu())

    def post_hook(module, args, output):
        act = output[:, -1, :].detach().to(torch.float32)
        projections.append((act @ direction).cpu())

    if act_name == "resid_pre":
        handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    else:
        handle = layer.register_forward_hook(post_hook)

    for i in range(0, len(instructions), batch_size):
        tokens = tokenize_instructions(tokenizer, instructions[i : i + batch_size]).to(device)
        with torch.no_grad():
            model(tokens)

    handle.remove()
    return torch.cat(projections)


def detect(model_path, direction_path, config, n_probe=32):
    payload = torch.load(direction_path, map_location="cpu")
    act_name, layer_idx, direction = payload["act_name"], payload["layer"], payload["direction"]

    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, device_map=config.device)
    model.eval()

    _, harmful_test = get_harmful_instructions()
    probe = harmful_test[:n_probe]

    proj = collect_projection(model, tokenizer, probe, act_name, layer_idx, direction)

    del model
    torch.cuda.empty_cache()

    return {
        "model_path": model_path,
        "act_name": act_name,
        "layer": layer_idx,
        "mean_projection": proj.mean().item(),
        "std_projection": proj.std().item(),
        "n": len(proj),
    }


def classify(baseline_stats, candidate_stats, threshold=0.2):
    baseline_mag = abs(baseline_stats["mean_projection"])
    candidate_mag = abs(candidate_stats["mean_projection"])
    ratio = candidate_mag / baseline_mag if baseline_mag > 1e-9 else float("nan")
    return ratio, ratio < threshold


def main():
    parser = argparse.ArgumentParser(prog="argos-detect")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--n-probe", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output", default="results/phase3_detect.json")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)

    baseline_stats = detect(args.baseline, args.direction, config, args.n_probe)
    candidate_stats = detect(args.candidate, args.direction, config, args.n_probe)
    ratio, flagged = classify(baseline_stats, candidate_stats, args.threshold)

    result = {
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "ratio": ratio,
        "flagged_as_ablated": flagged,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

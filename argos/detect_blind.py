import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from argos.activations import collect_activations
from argos.config import ArgosConfig
from argos.data import get_harmful_instructions, get_harmless_instructions
from argos.direction import compute_direction_norms


def collect_norm_profile(model_path, config):
    dtype = getattr(torch, config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, device_map=config.device)
    model.eval()

    harmful_train, _ = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions(test_size=config.n_test_instructions)

    harmful_acts, harmless_acts = collect_activations(model, tokenizer, harmful_train, harmless_train, config)
    norms = compute_direction_norms(harmful_acts, harmless_acts, config)

    del model
    torch.cuda.empty_cache()

    return {(act_name, layer_idx): norm for act_name, layer_idx, norm in norms}


def find_anomalous_layers(reference_profile, candidate_profile, drop_ratio=0.3, min_reference_norm=0.5):
    anomalies = []
    keys = sorted(reference_profile.keys(), key=lambda k: k[1])

    for i, key in enumerate(keys):
        ref_norm = reference_profile[key]
        cand_norm = candidate_profile.get(key)
        if cand_norm is None or ref_norm < min_reference_norm:
            continue

        ratio = cand_norm / ref_norm

        neighbors = [keys[j] for j in (i - 1, i + 1) if 0 <= j < len(keys)]
        neighbor_ratios = [
            candidate_profile[n] / reference_profile[n]
            for n in neighbors
            if reference_profile[n] >= min_reference_norm and candidate_profile.get(n) is not None
        ]
        local_baseline = sum(neighbor_ratios) / len(neighbor_ratios) if neighbor_ratios else 1.0

        if ratio < drop_ratio and ratio < local_baseline * drop_ratio:
            anomalies.append(
                {
                    "act_name": key[0],
                    "layer": key[1],
                    "reference_norm": ref_norm,
                    "candidate_norm": cand_norm,
                    "ratio": ratio,
                }
            )

    return anomalies


def main():
    parser = argparse.ArgumentParser(prog="argos-detect-blind")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--drop-ratio", type=float, default=0.3)
    parser.add_argument("--output", default="results/phase3_detect_blind.json")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)

    reference_profile = collect_norm_profile(args.reference, config)
    candidate_profile = collect_norm_profile(args.candidate, config)

    anomalies = find_anomalous_layers(reference_profile, candidate_profile, args.drop_ratio)

    result = {
        "reference": args.reference,
        "candidate": args.candidate,
        "reference_profile": {f"{k[0]}:{k[1]}": v for k, v in sorted(reference_profile.items(), key=lambda x: x[0][1])},
        "candidate_profile": {f"{k[0]}:{k[1]}": v for k, v in sorted(candidate_profile.items(), key=lambda x: x[0][1])},
        "anomalies": anomalies,
        "flagged_as_ablated": len(anomalies) > 0,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps({"anomalies": anomalies, "flagged_as_ablated": result["flagged_as_ablated"]}, indent=2))


if __name__ == "__main__":
    main()

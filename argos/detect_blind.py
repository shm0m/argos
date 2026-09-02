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


def find_anomalous_layers(reference_profile, candidate_profile, drop_ratio=0.7, min_reference_norm=0.5, baseline_layers=3):
    """Flague les couches dont le rapport candidat/reference chute bien en dessous
    du rapport observe sur les toutes premieres couches.

    L'ablation edite les poids de TOUTES les couches avec une seule direction (mesuree
    a une couche donnee) : son empreinte sur le profil de normes n'est donc pas un pic
    isole a cette couche, mais une baisse progressive qui s'installe et se maintient a
    partir d'un certain point du reseau. On compare chaque couche a une ligne de base
    "sans alteration" prise sur les premieres couches, plutot qu'a ses seuls voisins
    immediats.
    """
    keys = sorted(reference_profile.keys(), key=lambda k: k[1])
    valid = [
        k for k in keys if reference_profile[k] >= min_reference_norm and candidate_profile.get(k) is not None
    ]
    if not valid:
        return []

    ratios = {k: candidate_profile[k] / reference_profile[k] for k in valid}
    baseline_keys = valid[:baseline_layers]
    baseline = sum(ratios[k] for k in baseline_keys) / len(baseline_keys)

    anomalies = []
    for key in valid:
        if ratios[key] < baseline * drop_ratio:
            anomalies.append(
                {
                    "act_name": key[0],
                    "layer": key[1],
                    "reference_norm": reference_profile[key],
                    "candidate_norm": candidate_profile[key],
                    "ratio": ratios[key],
                    "baseline_ratio": baseline,
                }
            )

    return anomalies


def main():
    parser = argparse.ArgumentParser(prog="argos-detect-blind")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--drop-ratio", type=float, default=0.7)
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

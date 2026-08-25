import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from argos.ablate import apply_ablation
from argos.activations import collect_activations
from argos.config import ArgosConfig
from argos.data import get_harmful_instructions, get_harmless_instructions
from argos.direction import compute_refusal_directions, score_directions
from argos.eval.refusal import degenerate_rate, refusal_rate


def run(config: ArgosConfig, output_dir: str):
    dtype = getattr(torch, config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id, torch_dtype=dtype, device_map=config.device
    )
    model.eval()

    harmful_train, harmful_test = get_harmful_instructions()
    harmless_train, _ = get_harmless_instructions(test_size=config.n_test_instructions)

    harmful_acts, harmless_acts = collect_activations(
        model, tokenizer, harmful_train, harmless_train, config
    )
    directions = compute_refusal_directions(harmful_acts, harmless_acts, config)
    if not directions:
        raise RuntimeError("aucune direction de refus valide (toutes degenerees ou NaN)")

    evals = score_directions(model, tokenizer, directions, harmful_test, config)

    scores = [refusal_rate(gen) for gen in evals]
    degeneracy = [degenerate_rate(gen) for gen in evals]

    valid_indices = [i for i in range(len(evals)) if degeneracy[i] < 0.5]
    if not valid_indices:
        raise RuntimeError(
            "toutes les directions candidates produisent des generations degenerees "
            "(texte vide/trop court) : probable NaN en amont, ne pas selectionner"
        )

    best_idx = min(valid_indices, key=lambda i: scores[i])
    print(
        f"direction #{best_idx} selectionnee, taux de refus residuel = {scores[best_idx]:.2%}, "
        f"taux de generations degenerees = {degeneracy[best_idx]:.2%}"
    )

    apply_ablation(model, directions[best_idx])
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(prog="argos")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--output", default="results/ablated-model")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)
    run(config, args.output)


if __name__ == "__main__":
    main()

import argparse
import json

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from argos.config import ArgosConfig
from argos.data import get_harmful_instructions
from argos.direction import get_generations
from argos.eval.capability import run_capability_benchmarks
from argos.eval.refusal import refusal_stats

LIGHT_TASKS = [
    "hellaswag",
    "mmlu_abstract_algebra",
    "mmlu_high_school_mathematics",
    "mmlu_moral_scenarios",
    "mmlu_professional_law",
]


def measure(model_path, config, tasks, limit):
    dtype = getattr(torch, config.dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype, device_map=config.device)
    model.eval()

    _, harmful_test = get_harmful_instructions()
    generations = get_generations(model, tokenizer, harmful_test[: config.n_test_instructions])
    refusal = refusal_stats(generations)

    capability = run_capability_benchmarks(model, model_path, tasks=tasks, limit=limit) if tasks else {}

    del model
    torch.cuda.empty_cache()

    return {"model_path": model_path, "refusal": refusal, "capability": capability}


def main():
    parser = argparse.ArgumentParser(prog="argos-measure")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--ablated", required=True)
    parser.add_argument("--tasks", nargs="+", default=LIGHT_TASKS)
    parser.add_argument("--include-gsm8k", action="store_true")
    parser.add_argument("--gsm8k-limit", type=int, default=15)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="results/phase2_measure.json")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)

    results = {
        "baseline": measure(args.baseline, config, args.tasks, args.limit),
        "ablated": measure(args.ablated, config, args.tasks, args.limit),
    }

    if args.include_gsm8k:
        results["baseline_gsm8k"] = measure(args.baseline, config, ["gsm8k"], args.gsm8k_limit)
        results["ablated_gsm8k"] = measure(args.ablated, config, ["gsm8k"], args.gsm8k_limit)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


def run_capability_benchmarks(model, model_path, tasks=("hellaswag", "gsm8k", "mmlu_abstract_algebra"), limit=50, batch_size=8):
    lm = HFLM(pretrained=model, tokenizer=model_path, batch_size=batch_size)
    results = simple_evaluate(model=lm, tasks=list(tasks), limit=limit)
    return results["results"]

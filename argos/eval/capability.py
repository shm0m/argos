"""Mesure de la capacité de raisonnement (MMLU / GSM8K / HellaSwag) avant/après ablation (Phase 2).

Différenciateur du projet par rapport à l'article source : quantifier ce
compromis couche par couche plutôt que le corriger après coup par DPO.
"""


def run_capability_benchmarks(model_path: str, tasks: tuple[str, ...] = ("mmlu", "gsm8k", "hellaswag")):
    raise NotImplementedError("Phase 2 : wrapper autour de lm-eval-harness.")

"""Collecte des activations du residual stream (Phase 1)."""

from argos.config import ArgosConfig


def collect_activations(model, tokenizer, harmful_tokens, harmless_tokens, config: ArgosConfig):
    raise NotImplementedError("Phase 1 : run_with_cache sur les instructions nuisibles/bénignes.")

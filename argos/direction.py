"""Calcul et sélection de la direction de refus (Phase 1)."""

from argos.config import ArgosConfig


def compute_refusal_directions(harmful_acts, harmless_acts, config: ArgosConfig):
    raise NotImplementedError("Phase 1 : différence de moyennes normalisée, par couche.")


def score_directions(model, tokenizer, directions, test_instructions, config: ArgosConfig):
    raise NotImplementedError("Phase 1 : génération avec hook d'ablation par direction candidate.")

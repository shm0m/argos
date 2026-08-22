"""Chargement des jeux d'instructions nuisibles/bénignes (Phase 1)."""

from typing import TypedDict


class ChatTurn(TypedDict):
    role: str
    content: str


def reformat_texts(texts: list[str]) -> list[list[ChatTurn]]:
    return [[{"role": "user", "content": text}] for text in texts]


def get_harmful_instructions() -> tuple[list, list]:
    raise NotImplementedError("Phase 1 : charger depuis un benchmark public (AdvBench/HarmBench).")


def get_harmless_instructions() -> tuple[list, list]:
    raise NotImplementedError("Phase 1 : charger depuis un jeu d'instructions bénignes (ex. Alpaca).")

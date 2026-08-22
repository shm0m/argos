"""Point d'entrée CLI (Phase 4)."""

import argparse

from argos.config import ArgosConfig


def main() -> None:
    parser = argparse.ArgumentParser(prog="argos")
    parser.add_argument("--config", default="configs/ministral-3b.yaml")
    args = parser.parse_args()

    config = ArgosConfig.from_yaml(args.config)
    raise NotImplementedError(f"Phase 1+ : pipeline non encore implémenté (config chargée : {config.model_id}).")


if __name__ == "__main__":
    main()

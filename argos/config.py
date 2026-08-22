from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ArgosConfig:
    model_id: str
    dtype: str = "bfloat16"
    device: str = "cuda"
    activation_layers: list[str] = field(default_factory=lambda: ["resid_pre", "resid_mid", "resid_post"])
    selected_layers: list[str] = field(default_factory=lambda: ["resid_pre"])
    n_train_instructions: int = 256
    n_test_instructions: int = 32
    batch_size: int = 16
    eval_top_n: int = 20

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArgosConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

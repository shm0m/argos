from argos.config import ArgosConfig


def test_from_yaml_loads_ministral_config():
    config = ArgosConfig.from_yaml("configs/ministral-3b.yaml")
    assert config.model_id == "mistralai/Ministral-3-3B-Instruct-2512-BF16"
    assert "resid_pre" in config.activation_layers

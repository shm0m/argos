import random

from datasets import load_dataset


def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]


def get_harmful_instructions(test_size=32, seed=42):
    dataset = load_dataset("walledai/AdvBench", split="train")
    texts = list(dataset["prompt"])
    rng = random.Random(seed)
    rng.shuffle(texts)
    test = texts[:test_size]
    train = texts[test_size:]
    return reformat_texts(train), reformat_texts(test)


def get_harmless_instructions(test_size=32, seed=42):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    texts = [ex["instruction"] for ex in dataset if not ex["input"]]
    rng = random.Random(seed)
    rng.shuffle(texts)
    test = texts[:test_size]
    train = texts[test_size:]
    return reformat_texts(train), reformat_texts(test)

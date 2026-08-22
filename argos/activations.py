import gc
from collections import defaultdict

import torch
from tqdm import tqdm


def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def collect_activations(model, tokenizer, harmful_instructions, harmless_instructions, config):
    n = min(config.n_train_instructions, len(harmful_instructions), len(harmless_instructions))
    harmful_tokens = tokenize_instructions(tokenizer, harmful_instructions[:n])
    harmless_tokens = tokenize_instructions(tokenizer, harmless_instructions[:n])

    harmful = defaultdict(list)
    harmless = defaultdict(list)

    num_batches = (n + config.batch_size - 1) // config.batch_size
    for i in tqdm(range(num_batches)):
        start = i * config.batch_size
        end = min(n, start + config.batch_size)

        _, harmful_cache = model.run_with_cache(
            harmful_tokens[start:end],
            names_filter=lambda name: "resid" in name,
            device="cpu",
            reset_hooks_end=True,
        )
        _, harmless_cache = model.run_with_cache(
            harmless_tokens[start:end],
            names_filter=lambda name: "resid" in name,
            device="cpu",
            reset_hooks_end=True,
        )

        for key in harmful_cache:
            harmful[key].append(harmful_cache[key])
            harmless[key].append(harmless_cache[key])

        del harmful_cache, harmless_cache
        gc.collect()
        torch.cuda.empty_cache()

    harmful = {k: torch.cat(v) for k, v in harmful.items()}
    harmless = {k: torch.cat(v) for k, v in harmless.items()}
    return harmful, harmless

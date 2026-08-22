from collections import defaultdict

import torch
from tqdm import tqdm


def get_decoder(model):
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    return model.model


def get_layers(model):
    return get_decoder(model).layers


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

    device = next(model.parameters()).device
    layers = get_layers(model)

    harmful = {"resid_pre": defaultdict(list), "resid_post": defaultdict(list)}
    harmless = {"resid_pre": defaultdict(list), "resid_post": defaultdict(list)}

    def run(tokens_batch, store):
        handles = []
        for i, layer in enumerate(layers):
            def pre_hook(module, args, kwargs, i=i):
                store["resid_pre"][i].append(args[0][:, -1, :].detach().to("cpu", torch.float32))

            def post_hook(module, args, output, i=i):
                store["resid_post"][i].append(output[:, -1, :].detach().to("cpu", torch.float32))

            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(layer.register_forward_hook(post_hook))

        with torch.no_grad():
            model(tokens_batch.to(device))

        for h in handles:
            h.remove()

    num_batches = (n + config.batch_size - 1) // config.batch_size
    for i in tqdm(range(num_batches)):
        start = i * config.batch_size
        end = min(n, start + config.batch_size)
        run(harmful_tokens[start:end], harmful)
        run(harmless_tokens[start:end], harmless)

    for store in (harmful, harmless):
        for act_name in store:
            for layer_idx in list(store[act_name]):
                store[act_name][layer_idx] = torch.cat(store[act_name][layer_idx], dim=0)

    return harmful, harmless

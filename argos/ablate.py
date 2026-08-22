import torch

from argos.activations import get_decoder


def orthogonalize_matrix(matrix, direction, axis=-1):
    direction = direction.to(matrix.device, matrix.dtype)
    if axis == -1:
        proj = (matrix @ direction).unsqueeze(-1) * direction
    else:
        proj = torch.outer(direction, direction @ matrix)
    return matrix - proj


def apply_ablation(model, direction):
    decoder = get_decoder(model)
    decoder.embed_tokens.weight.data = orthogonalize_matrix(
        decoder.embed_tokens.weight.data, direction, axis=-1
    )
    for layer in decoder.layers:
        layer.self_attn.o_proj.weight.data = orthogonalize_matrix(
            layer.self_attn.o_proj.weight.data, direction, axis=0
        )
        layer.mlp.down_proj.weight.data = orthogonalize_matrix(
            layer.mlp.down_proj.weight.data, direction, axis=0
        )
    return model

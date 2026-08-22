import einops
from tqdm import tqdm


def orthogonalize_matrix(matrix, direction):
    proj = (
        einops.einsum(matrix, direction.view(-1, 1), "... d_model, d_model single -> ... single")
        * direction
    )
    return matrix - proj


def apply_ablation(model, direction):
    if direction.device != model.W_E.device:
        direction = direction.to(model.W_E.device)
    model.W_E.data = orthogonalize_matrix(model.W_E, direction)

    for block in tqdm(model.blocks):
        if direction.device != block.attn.W_O.device:
            direction = direction.to(block.attn.W_O.device)
        block.attn.W_O.data = orthogonalize_matrix(block.attn.W_O, direction)
        block.mlp.W_out.data = orthogonalize_matrix(block.mlp.W_out, direction)

    return model

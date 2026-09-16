import torch

window_size = 32
num_random = 2
num_global = 1

def bigbird_mask(block_size=256, window_size=32, num_random=2, num_global=1):
    masks = torch.zeros(block_size, block_size)

    # Local sliding-window attention
    for i in range(block_size):
        start = max(0, i - window_size + 1)
        masks[i, start:i + 1] = 1

    # Random attention
    for i in range(1, block_size):
        pos = torch.randperm(i)[:min(i, num_random)]
        for j in pos:
            masks[i, j] = 1

    # Global attention
    masks[:num_global] = 1
    masks[:, :num_global] = 1

    # Keep everything causal
    masks *= torch.tril(torch.ones(block_size, block_size))

    return masks

def causal_mask(block_size=256):
    return torch.tril(torch.ones(block_size, block_size))

def sliding_window_mask(block_size=256, window_size=32):
    masks = torch.zeros(block_size, block_size)
    for i in range(block_size):
        for j in range(max(0, i + 1 - window_size), i + 1):
            masks[i, j] = 1
    return masks
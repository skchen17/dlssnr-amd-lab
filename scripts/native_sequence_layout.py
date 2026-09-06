"""Recovered packed 1D sequence adapters, using 16-token/32-channel fragments."""
import torch


def sequence_indices(tokens, channels, *, device=None):
    if tokens <= 0 or channels <= 0 or channels % 32:
        raise ValueError('positive token count and channels divisible by32 required')
    t, c = torch.arange(tokens, device=device)[:, None], torch.arange(channels, device=device)[None, :]
    row, col = t % 16, c % 32
    mma = col // 16 * 2 + col % 16 // 8
    pairs = torch.tensor([0, 2, 1, 3, 4, 6, 5, 7], device=device)
    element = (row >= 8) * 2 + (col & 1)
    operation = pairs[mma * 2 + element // 2]
    return (t // 16 * (16 * channels) + c // 32 * 512
            + (row % 8 * 4 + col % 8 // 2) * 16 + operation * 2 + (element & 1)).long()


def unpack_sequence(packed, tokens, channels):
    padded = (tokens + 31) // 32 * 32
    if packed.ndim != 1 or packed.numel() != padded * channels:
        raise ValueError('incorrect packed sequence size; explicit padded region required')
    return packed[sequence_indices(tokens, channels, device=packed.device)].unsqueeze(0)


def pack_sequence(logical):
    if logical.ndim != 3 or logical.shape[0] != 1:
        raise ValueError('packed sequence adapter currently takes batch1')
    _, tokens, channels = logical.shape
    index = sequence_indices(tokens, channels, device=logical.device)
    packed = torch.zeros(((tokens + 31) // 32 * 32) * channels, device=logical.device, dtype=logical.dtype)
    packed[index] = logical[0]
    return packed

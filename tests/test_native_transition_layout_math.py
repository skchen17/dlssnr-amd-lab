import torch

from scripts.native_packed_swin import gather_packed, scatter_packed


PAIRS = (0, 2, 1, 3, 4, 6, 5, 7)


def packed_offset(y, x, channel, width, channels):
    row = y % 4 * 4 + x % 4
    col = channel % 32
    mma = col // 16 * 2 + col % 16 // 8
    element = (row >= 8) * 2 + (col & 1)
    operation = PAIRS[mma * 2 + element // 2]
    local = channel // 32 * 512 + (row % 8 * 4 + col % 8 // 2) * 16 + \
        operation * 2 + (element & 1)
    return (y // 4 * (width // 4) + x // 4) * (16 * channels) + local


def logical_from_physical(physical, width, channels, ox, oy):
    cell_elements = 16 * channels
    cell_x = physical // cell_elements % (width // 4)
    cell_y = physical // cell_elements // (width // 4)
    local = physical % cell_elements
    head, within = local // 512, local % 512
    lane, element = within // 16, within % 16
    tx, ty = cell_x - ox // 4, cell_y - oy // 4
    gx = (width - ox + 7) // 8
    window = ty // 2 * gx + tx // 2
    cell = ty % 2 * 2 + tx % 2
    row = lane // 4 + element % 8 // 4 * 8
    col = head * 32 + lane % 4 * 2 + element % 4 // 2 * 8 + \
        element // 8 * 16 + (element & 1)
    return (window * 64 + cell * 16 + row) * channels + col


def test_native_scatter_and_transition_inverse_match_reference_layout():
    width = height = 8
    channels = 32
    packed_seed = torch.arange(width * height * channels, dtype=torch.int64)
    for ox, oy in ((0, 0), (0, -4), (-4, 0), (-4, -4)):
        _, mapping = gather_packed(packed_seed, width, height, channels, ox, oy)
        windows = ((width - ox + 7) // 8) * ((height - oy + 7) // 8)
        logical = torch.arange(windows * 64 * channels, dtype=torch.int64).reshape(
            windows, 64, channels)
        packed = scatter_packed(logical, mapping, width, height)
        expected = torch.tensor([logical_from_physical(i, width, channels, ox, oy)
                                 for i in range(packed.numel())])
        assert torch.equal(packed, expected)
        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    physical = packed_offset(y, x, channel, width, channels)
                    assert expected[physical] == logical_from_physical(
                        physical, width, channels, ox, oy)

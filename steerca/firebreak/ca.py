import torch
import torch.nn.functional as F

DIRT = 0
TREE = 1
FIRE = 2

_KERNELS = {}

def _fire_kernel(device: torch.device) -> torch.Tensor:
    """3x3 Moore-neighborhood kernel, cached per device."""
    kernel = _KERNELS.get(device)
    if kernel is None:
        kernel = torch.tensor(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]],
            dtype=torch.float, device=device
        ).view(1, 1, 3, 3)  # Shape: (out_channels, in_channels, H, W)
        _KERNELS[device] = kernel
    return kernel

@torch.compile
def _step_impl(grid: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Update the grid state using forest fire rules."""
        current_fire = torch.unsqueeze(grid[..., FIRE, :, :], dim=-3)
        current_tree = torch.unsqueeze(grid[..., TREE, :, :], dim=-3)
        current_dirt = torch.unsqueeze(grid[..., DIRT, :, :], dim=-3)

        # Calculate FIRE neighbors with convolution
        fire_neighbors = F.conv2d(
            current_fire.float(),
            kernel,
            padding=1
        ) # Shape: (B, 1, H, W)

        # Apply state transitions
        # FIRE burns out to DIRT
        new_dirt = current_dirt | current_fire

        # Trees become FIRE if they have at least one FIRE neighbor
        new_fire = current_tree & (fire_neighbors >= 1)

        new_tree = current_tree & (fire_neighbors == 0)

        return torch.cat([new_dirt, new_tree, new_fire], dim=-3)

def step(grid: torch.Tensor) -> torch.Tensor:
        # The grid is a one-hot encoded tensor with shape (B, 3, H, W)
        # where B is the batch size, 3 corresponds to DIRT, TREE, FIRE
        # and H, W are the height and width of the grid.
        return _step_impl(grid, _fire_kernel(grid.device))

def fire_neighbors(grid: torch.Tensor) -> torch.Tensor:
        """Burning Moore neighbours of every cell, shape (..., H, W).

        Exactly the quantity the ignition rule thresholds -- a tree catches fire
        next tick iff this is >= 1 -- so anything that needs to ask "is this tree
        about to burn?" should ask here rather than reimplement the test and
        risk drifting from the rule that actually decides it.
        """
        fire = grid[..., FIRE, :, :].float()
        h, w = fire.shape[-2], fire.shape[-1]
        flat = fire.reshape(-1, 1, h, w)
        counts = F.conv2d(flat, _fire_kernel(grid.device), padding=1)
        return counts.reshape(*fire.shape[:-2], h, w)


def num_trees(grid: torch.Tensor):
        return grid[..., TREE, :, :].sum().item()

def num_fires(grid: torch.Tensor):
        return grid[..., FIRE, :, :].sum().item()

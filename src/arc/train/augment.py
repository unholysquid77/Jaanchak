"""GPU-side batch augmentation for the inner adaptation loop.

Needed because the inner objective requires a *second view* of each support image (see
``arc.train.objectives``), and round-tripping to PIL per inner step would dominate the
step cost. These operate on already-normalised CHW tensors on the device.

Deliberately minimal -- random resized crop and horizontal flip only. The augmentation
is part of the adaptation objective, not a trick to win: anything stronger (colour
jitter, blur) would change the representation-drift measurements we report, and would
have to be ablated separately.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_resized_crop_flip(
    x: torch.Tensor,
    scale: tuple[float, float] = (0.6, 1.0),
    hflip: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-sample random resized crop plus optional horizontal flip.

    Implemented as a single ``affine_grid`` / ``grid_sample`` so every image in the
    batch gets an independent crop in one kernel. Reflection padding keeps the sampler
    from inventing zero (i.e. mean-colour) borders when a crop runs off the edge.

    Parameters
    ----------
    scale
        Range of the *area* fraction retained by the crop, matching the convention of
        ``torchvision.transforms.RandomResizedCrop``.
    """
    if x.ndim != 4:
        raise ValueError(f"expected NCHW, got shape {tuple(x.shape)}")

    n = x.size(0)

    def uniform() -> torch.Tensor:
        # torch.rand takes a generator; torch.empty(...).uniform_ does not.
        return torch.rand(n, device=x.device, generator=generator)

    # Side-length fraction is the square root of the area fraction.
    side = (uniform() * (scale[1] - scale[0]) + scale[0]).sqrt()

    # Centre the crop anywhere it still fits inside the image.
    max_shift = 1.0 - side
    tx = (uniform() * 2 - 1) * max_shift
    ty = (uniform() * 2 - 1) * max_shift

    sx = side.clone()
    if hflip:
        sx = sx * torch.where(uniform() < 0.5, -1.0, 1.0)

    theta = torch.zeros(n, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = sx
    theta[:, 1, 1] = side
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

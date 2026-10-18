"""Reconstruction metrics required by the GIA experiment."""

from __future__ import annotations

import torch


def total_variation(images: torch.Tensor) -> torch.Tensor:
    dx = torch.mean(torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:]))
    dy = torch.mean(torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :]))
    return dx + dy


def mse(images: torch.Tensor, reference: torch.Tensor) -> float:
    return torch.mean((images.detach() - reference.detach()) ** 2).item()


def psnr(
    images: torch.Tensor, reference: torch.Tensor, data_range: float = 1.0
) -> float:
    values = (images.detach() - reference.detach()).square().flatten(1).mean(1)
    scores = torch.where(
        values == 0,
        torch.full_like(values, float("inf")),
        10.0 * torch.log10((data_range**2) / values),
    )
    return scores.mean().item()


def ssim(
    images: torch.Tensor, reference: torch.Tensor, data_range: float = 1.0
) -> float:
    """Compute a dependency-free global SSIM value, averaged over the batch."""
    images = images.detach().flatten(1)
    reference = reference.detach().flatten(1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = images.mean(dim=1)
    mu_y = reference.mean(dim=1)
    var_x = images.var(dim=1, unbiased=False)
    var_y = reference.var(dim=1, unbiased=False)
    covariance = ((images - mu_x[:, None]) * (reference - mu_y[:, None])).mean(dim=1)
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return score.mean().item()


def cw_ssim(
    images: torch.Tensor,
    reference: torch.Tensor,
    scales: int = 5,
    stabilizer: float = 1e-6,
) -> float | None:
    """Compute the complex-wavelet SSIM used by the source GIA repository."""
    try:
        from pytorch_wavelets import DTCWTForward
    except ImportError:
        return None
    transform = DTCWTForward(
        J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=True
    ).to(device=images.device, dtype=images.dtype)
    image_coefficients = transform(images)
    reference_coefficients = transform(reference)
    score = images.new_zeros(images.shape[0])
    used_scales = 0
    for image_scale, reference_scale in zip(
        image_coefficients[1], reference_coefficients[1]
    ):
        if len(image_scale) == 0:
            continue
        image_complex = torch.view_as_complex(image_scale)
        reference_complex = torch.view_as_complex(reference_scale)
        conjugate = (image_complex * reference_complex.conj()).sum(dim=2).abs()
        image_square = (image_complex * image_complex.conj()).abs().sum(dim=2)
        reference_square = (
            (reference_complex * reference_complex.conj()).abs().sum(dim=2)
        )
        value = (2 * conjugate + stabilizer) / (
            image_square + reference_square + stabilizer
        )
        score = score + value.mean(dim=(1, 2, 3))
        used_scales += 1
    return (score / used_scales).mean().item() if used_scales else None

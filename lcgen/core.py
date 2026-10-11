"""An explicit interpretation of LCGen, not the unavailable author implementation.

See docs/formula_ambiguities.md before interpreting experimental results.
All probabilities here are elementwise Gaussian *densities*, not joint products.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class CertaintyOptions:
    gamma: float = 10.0
    mean_mode: str = "paper"
    gradient_sign: str = "density"
    angle_units: str = "radians"

    def __post_init__(self):
        if self.gamma <= 0 or not math.isfinite(self.gamma):
            raise ValueError("gamma must be finite and positive")
        if self.mean_mode not in {"paper", "ddpm"}:
            raise ValueError("mean_mode must be paper or ddpm")
        if self.gradient_sign not in {"density", "paper"}:
            raise ValueError("gradient_sign must be density or paper")
        if self.angle_units not in {"radians", "degrees"}:
            raise ValueError("angle_units must be radians or degrees")


@dataclass
class CertaintyResult:
    gradient: Tensor
    density: Tensor
    weighted_density: Tensor
    residual: Tensor


def view_weight(azimuth_degrees: Tensor, units: str = "radians") -> Tensor:
    """threestudio's +x/front is 0 degrees; wrap evaluation's [0,360) too."""
    wrapped = torch.remainder(azimuth_degrees + 180.0, 360.0) - 180.0
    if units == "radians":
        wrapped = torch.deg2rad(wrapped)
    elif units != "degrees":
        raise ValueError("units must be radians or degrees")
    return wrapped.abs()


def _batch_view(values: Tensor, sample: Tensor) -> Tensor:
    return values.reshape(-1, *([1] * (sample.ndim - 1)))


def transition_mean(x_t: Tensor, predicted_noise: Tensor, beta: Tensor,
                    alpha_bar: Tensor, mode: str) -> Tensor:
    alpha = 1.0 - beta
    if mode == "paper":
        # Literal Eq. (8): (x_t - sqrt(1-beta_t)*epsilon_hat)/sqrt(1-beta_t).
        return x_t / alpha.sqrt() - predicted_noise
    if mode == "ddpm":
        # A separately named sensitivity variant, NOT silently substituted for Eq. (8).
        return (x_t - beta / (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha.sqrt()
    raise ValueError("unknown transition mean")


def gaussian_density(x_previous: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
    # Log evaluation avoids intermediate divisions/overflow. Very small densities
    # may legitimately underflow to zero; we report this, not clamp them upward.
    return torch.exp(-0.5 * ((x_previous - mean).square() / variance
                            + torch.log(2.0 * math.pi * variance)))


@torch.no_grad()
def certainty_gradient(latents: Tensor, x_t: Tensor, predicted_noise: Tensor,
                       timesteps: Tensor, betas: Tensor, alphas_cumprod: Tensor,
                       azimuth_degrees: Tensor,
                       options: CertaintyOptions = CertaintyOptions()) -> CertaintyResult:
    """Return dL_cert/d(rendered clean latent), excluding the external strength.

    x_{t-1} is a DETACHED forward-noised reference, using the same epsilon as x_t.
    The predicted noise is detached, matching the SDS stop-gradient convention.
    L_cert is summed over coordinates and averaged over batch in the surrogate.
    This function returns the per-sample gradient before that batch average.
    """
    if latents.shape != x_t.shape or x_t.shape != predicted_noise.shape:
        raise ValueError("latents, x_t and predicted_noise must have identical shapes")
    if latents.ndim < 2 or timesteps.shape != (latents.shape[0],):
        raise ValueError("expected batched latents and one timestep per sample")
    if azimuth_degrees.shape != timesteps.shape:
        raise ValueError("expected one azimuth per sample")
    if bool((timesteps <= 0).any()) or bool((timesteps >= len(betas)).any()):
        raise ValueError("certainty requires 1 <= t < num_train_timesteps")
    # Keep densities out of float16, while preserving float64 for gradient tests.
    dtype = torch.float64 if latents.dtype == torch.float64 else torch.float32
    clean = latents.detach().to(dtype)
    noisy = x_t.detach().to(dtype)
    prediction = predicted_noise.detach().to(dtype)
    beta = _batch_view(betas.to(clean)[timesteps], clean)
    alpha_bar = _batch_view(alphas_cumprod.to(clean)[timesteps], clean)
    alpha_bar_previous = _batch_view(alphas_cumprod.to(clean)[timesteps - 1], clean)
    if not bool(((beta > 0) & (beta < 1)).all()):
        raise ValueError("betas must lie strictly between zero and one")
    epsilon = (noisy - alpha_bar.sqrt() * clean) / (1.0 - alpha_bar).sqrt()
    previous = (alpha_bar_previous.sqrt() * clean
                + (1.0 - alpha_bar_previous).sqrt() * epsilon).detach()
    mean = transition_mean(noisy, prediction, beta, alpha_bar, options.mean_mode)
    residual = previous - mean
    density = gaussian_density(previous, mean, beta)
    guidance = _batch_view(view_weight(azimuth_degrees.to(clean), options.angle_units), clean)
    weighted = density * guidance / options.gamma
    # dN(y;mu,sigma²)/dmu = +(y-mu)/sigma² * N. Appendix D prints minus.
    sign = 1.0 if options.gradient_sign == "density" else -1.0
    gradient = sign * residual / beta * weighted / (1.0 - beta).sqrt()
    # x_t = sqrt(alpha_bar_t)*rendered_latent + sqrt(1-alpha_bar_t)*epsilon.
    gradient = gradient * alpha_bar.sqrt()
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("non-finite certainty gradient")
    return CertaintyResult(gradient, density, weighted, residual)

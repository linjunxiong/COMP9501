"""Add LCGen to upstream SDS without another U-Net call or random draw."""
from dataclasses import dataclass
import math

import torch

import threestudio
from threestudio.models.guidance.stable_diffusion_guidance import StableDiffusionGuidance
from lcgen.core import CertaintyOptions, certainty_gradient


@threestudio.register("lcgen-stable-diffusion-guidance")
class LCGenStableDiffusionGuidance(StableDiffusionGuidance):
    @dataclass
    class Config(StableDiffusionGuidance.Config):
        cert_strength: float = 1.0
        cert_gamma: float = 10.0
        cert_mean_mode: str = "paper"
        cert_gradient_sign: str = "density"
        cert_angle_units: str = "radians"

    def configure(self):
        if self.cfg.use_sjc or self.cfg.use_img_loss:
            raise ValueError("this reproduction supports latent SDS only")
        if self.cfg.cert_strength < 0 or not math.isfinite(self.cfg.cert_strength):
            raise ValueError("cert_strength must be finite and nonnegative")
        self.cert_options = CertaintyOptions(
            gamma=self.cfg.cert_gamma,
            mean_mode=self.cfg.cert_mean_mode,
            gradient_sign=self.cfg.cert_gradient_sign,
            angle_units=self.cfg.cert_angle_units,
        )
        self._cert_stats = {}
        super().configure()

    def compute_grad_sds(self, latents, image, t, prompt_utils,
                         elevation, azimuth, camera_distances):
        grad, grad_img, info = super().compute_grad_sds(
            latents, image, t, prompt_utils, elevation, azimuth, camera_distances
        )
        # In particular, strength=0 follows the identical upstream path and RNG state.
        if self.cfg.cert_strength == 0:
            self._cert_stats = {}
            return grad, grad_img, info
        result = certainty_gradient(
            latents, info["latents_noisy"], info["noise_pred"], t,
            self.scheduler.betas, self.scheduler.alphas_cumprod,
            azimuth, self.cert_options,
        )
        extra = self.cfg.cert_strength * result.gradient
        self._cert_stats = {
            "cert_density_mean": result.density.mean(),
            "cert_objective": result.weighted_density.flatten(1).sum(1).mean(),
            "cert_grad_norm": extra.norm(),
            "sds_grad_norm_before_cert": grad.norm(),
            "cert_to_sds_grad_ratio": extra.norm() / grad.norm().clamp_min(1e-12),
            "cert_nonzero_fraction": (extra != 0).float().mean(),
            "cert_residual_rms": result.residual.square().mean().sqrt(),
        }
        # Upstream __call__ injects this via a detached-target surrogate loss.
        return grad + extra, grad_img, info

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        out.update(self._cert_stats)
        return out

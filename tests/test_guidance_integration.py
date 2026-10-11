"""Exercise the actual integration class against an isolated deterministic SDS base."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import torch

from lcgen.core import certainty_gradient


def load_integration(monkeypatch):
    registry = types.ModuleType("threestudio")
    registry.register = lambda name: lambda cls: cls
    base_module = types.ModuleType("threestudio.models.guidance.stable_diffusion_guidance")

    class FakeSDS:
        @dataclass
        class Config:
            use_sjc: bool = False
            use_img_loss: bool = False

        def configure(self):
            pass

        def compute_grad_sds(self, latents, image, t, prompt_utils, elevation, azimuth, distances):
            abar = self.scheduler.alphas_cumprod[t].reshape(-1,1)
            noisy = abar.sqrt()*latents.detach() + (1-abar).sqrt()*0.1
            return torch.full_like(latents,0.25), None, {
                "latents_noisy":noisy, "noise_pred":torch.full_like(latents,0.2)}

        def __call__(self, latents, t, azimuth):
            grad,_,_ = self.compute_grad_sds(latents,None,t,None,azimuth,azimuth,azimuth)
            return {"base_gradient":grad}

    base_module.StableDiffusionGuidance = FakeSDS
    monkeypatch.setitem(sys.modules,"threestudio",registry)
    monkeypatch.setitem(sys.modules,base_module.__name__,base_module)
    filename = Path(__file__).resolve().parents[1]/"lcgen/guidance.py"
    spec = importlib.util.spec_from_file_location("lcgen_integration_under_test",filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module)
    return module.LCGenStableDiffusionGuidance, FakeSDS


def instance(cls, strength):
    model = cls()
    model.cfg = cls.Config(cert_strength=strength)
    model.configure()
    betas = torch.tensor([0.05,0.1,0.2])
    model.scheduler = types.SimpleNamespace(betas=betas,alphas_cumprod=(1-betas).cumprod(0))
    return model


def test_zero_strength_matches_upstream_and_rng(monkeypatch):
    cls, base = load_integration(monkeypatch)
    model = instance(cls,0)
    x,t,angles = torch.tensor([[0.4]]),torch.tensor([1]),torch.tensor([90.])
    state = torch.random.get_rng_state().clone()
    actual = model(x,t,angles)
    expected,_,_ = base.compute_grad_sds(model,x,None,t,None,angles,angles,angles)
    torch.testing.assert_close(actual["base_gradient"],expected,rtol=0,atol=0)
    assert torch.equal(state,torch.random.get_rng_state())
    assert "cert_grad_norm" not in actual


def test_additive_gradient_and_diagnostics(monkeypatch):
    cls,base = load_integration(monkeypatch)
    model = instance(cls,2)
    x,t,angles = torch.tensor([[0.4]]),torch.tensor([1]),torch.tensor([90.])
    upstream,_,info = base.compute_grad_sds(model,x,None,t,None,angles,angles,angles)
    extra = certainty_gradient(x,info["latents_noisy"],info["noise_pred"],t,
                              model.scheduler.betas,model.scheduler.alphas_cumprod,
                              angles,model.cert_options).gradient
    actual = model(x,t,angles)
    torch.testing.assert_close(actual["base_gradient"],upstream+2*extra)
    assert actual["cert_grad_norm"] > 0
    assert actual["cert_nonzero_fraction"] == 1
    assert not any(key.startswith("loss_cert") for key in actual)

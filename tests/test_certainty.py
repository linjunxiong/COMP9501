import math

import pytest
import torch

from lcgen.core import CertaintyOptions, certainty_gradient, view_weight


@pytest.mark.parametrize("mode", ["paper", "ddpm"])
def test_gradient_matches_finite_difference_with_fixed_reference(mode):
    # Independent scalar objective: freeze x_{t-1} and epsilon_hat exactly as SDS does.
    clean = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    epsilon = torch.tensor([[0.1, 0.3]], dtype=torch.float64)
    prediction = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    betas = torch.tensor([0.05, 0.1, 0.2], dtype=torch.float64)
    bars = (1 - betas).cumprod(0)
    t = torch.tensor([1])
    a = float(bars[1])
    previous = math.sqrt(float(bars[0])) * clean + math.sqrt(1 - float(bars[0])) * epsilon
    noisy = math.sqrt(a) * clean + math.sqrt(1 - a) * epsilon
    actual = certainty_gradient(clean, noisy, prediction, t, betas, bars,
                                torch.tensor([90.0]), CertaintyOptions(mean_mode=mode)).gradient

    def objective(candidate):
        xt = math.sqrt(a) * candidate + math.sqrt(1-a) * epsilon
        if mode == "paper":
            mu = xt / math.sqrt(0.9) - prediction
        else:
            mu = (xt - 0.1 / math.sqrt(1-a) * prediction) / math.sqrt(0.9)
        pdf = torch.exp(-((previous-mu)**2)/(2*0.1)) / math.sqrt(2*math.pi*0.1)
        return pdf.sum() * (math.pi/2) / 10

    expected = torch.zeros_like(clean)
    for i in range(2):
        plus, minus = clean.clone(), clean.clone()
        plus[0, i] += 1e-6
        minus[0, i] -= 1e-6
        expected[0, i] = (objective(plus)-objective(minus))/(2e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9)
    assert objective(clean - 0.001*actual) < objective(clean)


def test_paper_sign_is_explicit_opposite_and_front_weight_is_zero():
    clean = torch.tensor([[0.4], [0.4]])
    beta = torch.tensor([0.05, 0.1, 0.2])
    bars = (1-beta).cumprod(0)
    t = torch.tensor([1, 1])
    noisy = bars[1].sqrt()*clean + (1-bars[1]).sqrt()*0.1
    kwargs = dict(latents=clean, x_t=noisy, predicted_noise=torch.full_like(clean, 0.2),
                  timesteps=t, betas=beta, alphas_cumprod=bars,
                  azimuth_degrees=torch.tensor([0.,180.]))
    corrected = certainty_gradient(**kwargs)
    printed = certainty_gradient(**kwargs, options=CertaintyOptions(gradient_sign="paper"))
    assert corrected.gradient[0].item() == 0
    assert corrected.gradient[1].abs().item() > 0
    torch.testing.assert_close(corrected.gradient, -printed.gradient)


def test_angle_wrap_and_units():
    angles = torch.tensor([0.,360.,-360.,90.,270.,180.,-180.])
    weights = view_weight(angles)
    torch.testing.assert_close(weights, torch.tensor([0.,0.,0.,math.pi/2,math.pi/2,math.pi,math.pi]))
    torch.testing.assert_close(view_weight(angles, "degrees"), weights*180/math.pi)


def test_no_random_draw_and_no_unet_gradient():
    clean = torch.tensor([[0.4]], requires_grad=True)
    prediction = torch.tensor([[0.2]], requires_grad=True)
    beta = torch.tensor([0.05, 0.1])
    bars = (1-beta).cumprod(0)
    state = torch.random.get_rng_state().clone()
    result = certainty_gradient(clean, torch.tensor([[0.3]]), prediction, torch.tensor([1]),
                                beta, bars, torch.tensor([90.]))
    assert not result.gradient.requires_grad
    assert clean.grad is None and prediction.grad is None
    assert torch.equal(state, torch.random.get_rng_state())


def test_float16_density_is_computed_in_float32_and_underflow_is_visible():
    x = torch.tensor([[0., 1.]], dtype=torch.float16)
    result = certainty_gradient(x, x, torch.full_like(x, 100.), torch.tensor([1]),
                                torch.tensor([.0001,.0002]), torch.tensor([.9999,.9997]),
                                torch.tensor([180.]))
    assert result.gradient.dtype == torch.float32
    assert torch.isfinite(result.gradient).all()
    assert result.density.eq(0).all()  # Do not manufacture certainty via a minimum clamp.


def test_detached_target_surrogate_injects_exact_additive_gradient():
    z = torch.tensor([[0.1,0.2],[0.3,0.4]], requires_grad=True)
    desired = torch.tensor([[0.7,-0.2],[0.3,-0.9]])
    target = (z-desired).detach()
    loss = 0.5*((z-target)**2).sum()/len(z)
    loss.backward()
    torch.testing.assert_close(z.grad, desired/len(z))


@pytest.mark.parametrize("options", [{"gamma":0}, {"mean_mode":"invalid"},
                                    {"gradient_sign":"invalid"}, {"angle_units":"invalid"}])
def test_invalid_options_fail(options):
    with pytest.raises(ValueError):
        CertaintyOptions(**options)

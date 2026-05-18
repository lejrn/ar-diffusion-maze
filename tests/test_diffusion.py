import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from diffusion import LinearNoiseSchedule, PathDenoiser


def test_q_sample_shape():
    sched = LinearNoiseSchedule(T=100)
    x0 = torch.randn(4, 20, 2)
    t = torch.randint(0, 100, (4,))
    xt, noise = sched.q_sample(x0, t)
    assert xt.shape == x0.shape
    assert noise.shape == x0.shape


def test_denoiser_output_shape():
    model = PathDenoiser(path_len=20)
    x = torch.randn(4, 20, 2)
    t = torch.randint(0, 100, (4,))
    out = model(x, t)
    assert out.shape == (4, 20, 2)


def test_alpha_bar_decreasing():
    sched = LinearNoiseSchedule(T=100)
    assert sched.alpha_bar[0] > sched.alpha_bar[-1]
    assert (sched.alpha_bar > 0).all()

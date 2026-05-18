import torch
import torch.nn as nn


class LinearNoiseSchedule:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise, noise


class PathDenoiser(nn.Module):
    def __init__(self, path_len=20, hidden=256, T=100):
        super().__init__()
        self.path_len = path_len
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(path_len * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, path_len * 2),
        )

    def forward(self, x, t):
        B = x.shape[0]
        t_emb = t.float().unsqueeze(1) / self.T
        x_flat = x.view(B, -1)
        inp = torch.cat([x_flat, t_emb], dim=1)
        return self.net(inp).view(B, self.path_len, 2)

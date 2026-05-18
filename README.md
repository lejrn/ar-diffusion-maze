# Toy Diffusion Policy — 2D Maze

Trains a DDPM (Denoising Diffusion Probabilistic Model) to generate
collision-free paths through a random 2D grid maze. The PathDenoiser
(MLP) learns to predict added Gaussian noise given a noisy path and
timestep. After training, DDPM reverse sampling generates new paths
from pure noise.

## Install

```bash
uv sync
```

## Train (~2 min on CPU)

```bash
uv run python src/train.py
```

## Visualize

```bash
uv run python src/visualize.py
# Opens: generated_path.png
```

## Test

```bash
uv run pytest tests/ -v
```

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diffusion import LinearNoiseSchedule, PathDenoiser

PATH_LEN = 20
MAZE_SIZE = 10


@torch.no_grad()
def ddpm_sample(model, schedule, n=1):
    x = torch.randn(n, PATH_LEN, 2)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        pred_noise = model(x, t_batch)
        ab = schedule.alpha_bar[t]
        x0_pred = (x - (1 - ab).sqrt() * pred_noise) / ab.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        if t > 0:
            ab_prev = schedule.alpha_bar[t - 1]
            sigma = ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            x = (
                ab_prev.sqrt() * x0_pred
                + (1 - ab_prev - sigma**2).sqrt() * pred_noise
                + sigma * torch.randn_like(x)
            )
        else:
            x = x0_pred

    return x


def render(checkpoint="checkpoint.pt", out="generated_path.png"):
    from maze import GridMaze

    ckpt = torch.load(checkpoint, weights_only=True)
    maze = GridMaze(size=ckpt["maze_size"], seed=ckpt["maze_seed"])
    schedule = LinearNoiseSchedule(T=100)
    model = PathDenoiser(path_len=PATH_LEN)
    model.load_state_dict(ckpt["model"])
    model.eval()

    path = ddpm_sample(model, schedule, n=1)[0].numpy()
    path = (path + 1) / 2.0 * (maze.size - 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.zeros((maze.size, maze.size, 3))
    grid[maze.walls] = [0.2, 0.2, 0.2]
    grid[~maze.walls] = [0.95, 0.95, 0.95]
    ax.imshow(grid, origin="upper")
    ax.plot(
        path[:, 1],
        path[:, 0],
        "r-o",
        markersize=4,
        linewidth=1.5,
        label="generated path",
    )
    ax.scatter([0], [0], c="green", s=80, zorder=5, label="start")
    ax.scatter(
        [maze.size - 1], [maze.size - 1], c="blue", s=80, zorder=5, label="goal"
    )
    ax.legend(fontsize=9)
    ax.set_title("DDPM-generated maze path")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    render()

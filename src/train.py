import numpy as np
import torch

from diffusion import LinearNoiseSchedule, PathDenoiser
from maze import GridMaze

PATH_LEN = 20
MAZE_SIZE = 10


def pad_path(path, size, length=PATH_LEN):
    arr = np.array(path, dtype=np.float32)
    if len(arr) < length:
        arr = np.pad(arr, ((0, length - len(arr)), (0, 0)), mode="edge")
    else:
        arr = arr[:length]
    return arr / (size - 1) * 2.0 - 1.0


def generate_dataset(maze, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    paths = [
        pad_path(p, maze.size)
        for _ in range(n * 2)
        if (p := maze.random_path(rng)) is not None
    ][:n]
    return torch.tensor(np.stack(paths))


def train(epochs=300, batch=64, lr=2e-3):
    maze = GridMaze(size=MAZE_SIZE)
    schedule = LinearNoiseSchedule(T=100)
    model = PathDenoiser(path_len=PATH_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = generate_dataset(maze)
    print(f"Dataset: {len(dataset)} paths")

    for epoch in range(epochs):
        perm = torch.randperm(len(dataset))
        total_loss = 0.0
        steps = 0

        for i in range(0, len(dataset) - batch, batch):
            x0 = dataset[perm[i : i + batch]]
            t = torch.randint(0, schedule.T, (len(x0),))
            xt, noise = schedule.q_sample(x0, t)
            loss = ((model(xt, t) - noise) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1

        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss={total_loss / steps:.4f}")

    torch.save(
        {"model": model.state_dict(), "maze_seed": 42, "maze_size": MAZE_SIZE},
        "checkpoint.pt",
    )
    print("Saved checkpoint.pt")
    return model, maze, schedule


if __name__ == "__main__":
    train()

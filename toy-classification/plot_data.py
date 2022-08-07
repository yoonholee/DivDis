#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from task_generator import generate_data
from utils import savefig

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs("figures/data", exist_ok=True)


def plot_data(tr_x, tr_y, te_x, num_train=15, num_test=30):
    tr_g = tr_x[tr_y.flatten() == 0][:num_train]
    plt.scatter(
        tr_g[:, 0],
        tr_g[:, 1],
        marker="s",
        zorder=10,
        s=50,
        c="firebrick",
        edgecolors="k",
        linewidth=1,
    )
    tr_g = tr_x[tr_y.flatten() == 1][:num_train]
    plt.scatter(
        tr_g[:, 0],
        tr_g[:, 1],
        marker="^",
        zorder=10,
        s=70,
        c="royalblue",
        edgecolors="k",
        linewidth=1,
    )

    plt.scatter(
        te_x[:num_test, 0],
        te_x[:num_test, 1],
        zorder=0,
        s=30,
        c="silver",
        edgecolors="k",
        linewidth=1,
    )


#%%
tr_x, tr_y = generate_data(1000, train=True)
te_x, _ = generate_data(1000, train=False)

plt.figure(figsize=(5, 5))
plot_data(tr_x, tr_y, te_x)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
savefig(f"data/default")

# %%

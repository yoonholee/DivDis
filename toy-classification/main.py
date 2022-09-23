#%%
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from task_generator import generate_data, sample_minibatch
from utils import savefig

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from divdis import DivDisLoss

SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

batch_size = 32
test_batch_size = 100
train_iter = 1000
log_every = 100
heads = 2
mode, reduction, aux_weight = "mi", "mean", 1.0

# For this toy dataset, directly maximizing L1 distance works too!
# mode, reduction, aux_weight = "l1", "mean", 0.1

exp_name = f"h{heads}_{mode}-{reduction}_w{aux_weight}_{SEED}"
os.makedirs(f"figures/temp/{exp_name}", exist_ok=True)

fig_save_times = sorted(
    [1, 2, 3, 4, 8, 16, 32, 64, 120, 128] + [50 * n for n in range(200)]
)
fig_save_times = [t for t in fig_save_times if t < train_iter]

training_data = generate_data(500, train=True)
test_data = generate_data(5000, train=False)
net = nn.Sequential(
    nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, heads)
)
opt = torch.optim.Adam(net.parameters())
loss_fn = DivDisLoss(heads=heads, mode=mode, reduction=reduction)


def plot(time=""):
    N = 20
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    xv, yv = np.meshgrid(x, y)
    inpt = torch.tensor(np.stack([xv.reshape(-1), yv.reshape(-1)], axis=-1)).float()
    with torch.no_grad():
        preds = net(inpt).reshape(N, N, heads).sigmoid().cpu()

    tr_x, tr_y = training_data
    for i in range(heads):
        plt.figure(figsize=(4, 4))
        plt.contourf(xv, yv, preds[:, :, i], cmap="RdBu", alpha=0.75)
        for g, c in [(0, "#E7040F"), (1, "#00449E")]:
            tr_g = tr_x[tr_y.flatten() == g]
            plt.scatter(tr_g[:, 0], tr_g[:, 1], zorder=10, s=10, c=c, edgecolors="k")
        plt.xlim(-1.0, 1.0)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        savefig(f"temp/{exp_name}/{time}_h{i}", transparent=True)


#%%
metrics = defaultdict(list)
for t in range(train_iter):
    x, y = sample_minibatch(training_data, batch_size)
    logits = net(x)
    logits_chunked = torch.chunk(logits, heads, dim=-1)
    losses = [F.binary_cross_entropy_with_logits(logit, y) for logit in logits_chunked]
    xent = sum(losses)

    target_x, target_y = sample_minibatch(test_data, test_batch_size)
    target_logits = net(target_x)
    repulsion_loss = loss_fn(target_logits)

    full_loss = xent + aux_weight * repulsion_loss
    opt.zero_grad()
    full_loss.backward()
    opt.step()

    for i in range(heads):
        corrects_i = (target_logits[:, i] > 0) == target_y.flatten()
        acc_i = corrects_i.float().mean()
        metrics[f"acc_{i}"].append(acc_i.item())
    metrics[f"xent"].append(xent.item())
    metrics[f"repulsion_loss"].append(repulsion_loss.item())

    if t in fig_save_times:
        print(f"Generating plots for {t}/{train_iter}")
        plot(t)

    if t % log_every == 0:
        print(f"{t} xent {xent.item():.5f} aux {repulsion_loss.item():.5f}")

#%% Train single ERM model (for comparison in learning curve)
net = nn.Sequential(
    nn.Linear(2, 40), nn.ReLU(), nn.Linear(40, 40), nn.ReLU(), nn.Linear(40, heads)
)
opt = torch.optim.Adam(net.parameters())

for t in range(train_iter):
    x, y = sample_minibatch(training_data, batch_size)
    logits = net(x)
    logits_chunked = torch.chunk(logits, heads, dim=-1)
    losses = [F.binary_cross_entropy_with_logits(logit, y) for logit in logits_chunked]
    full_loss = sum(losses)
    opt.zero_grad()
    full_loss.backward()
    opt.step()

    target_x, target_y = sample_minibatch(test_data, test_batch_size)
    target_logits = net(target_x)
    for i in range(heads):
        corrects_i = (target_logits[:, i] > 0) == target_y.flatten()
        acc_i = corrects_i.float().mean()
        metrics[f"ERM_acc_{i}"].append(acc_i.item())
        print(acc_i.item())
    if t % log_every == 0:
        print(f"{t} xent {xent.item():.5f}")

#%% Draw learning curves
def draw_full_curve(t=None, with_erm=False):
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))
    N = 10
    uniform = np.ones(N) / N
    axs[0].set_xlim(-10, 1000)
    axs[0].set_ylim(0.45, 1.05)
    smooth = lambda x: np.convolve(x, uniform, mode="valid")
    for i in [0, 1]:
        axs[0].plot(smooth(metrics[f"acc_{i}"]), alpha=0.8, linewidth=2)
    if with_erm:
        axs[0].plot(smooth(metrics["ERM_acc_0"]), c="dimgray", alpha=0.5, linewidth=2)
    axs[1].plot(smooth(metrics["xent"]), c="dimgray")
    axs[2].plot(smooth(metrics["repulsion_loss"]) * 50, c="dimgray")
    axs[0].set_ylabel("Accuracy")
    axs[1].set_ylabel("Cross-Entropy")
    axs[2].set_ylabel("MI")
    for ax in axs:
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_linewidth(1.2)
        ax.xaxis.set_tick_params(width=1.2)
        ax.yaxis.set_tick_params(width=1.2)
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
    if t:
        for ax in axs:
            ax.axvline(x=t, c="k")


draw_full_curve()
savefig(f"temp/{exp_name}/learning_curve_full")

draw_full_curve(with_erm=True)
savefig(f"temp/{exp_name}/learning_curve_full_with_ERM")

for t in fig_save_times:
    draw_full_curve(t=t)
    savefig(f"temp/{exp_name}/learning_curve_full_{t}")

plt.figure(figsize=(8, 2))
N = 10
uniform = np.ones(N) / N
plt.ylim(0.45, 1.05)
smooth = lambda x: np.convolve(x, uniform, mode="valid")
ax = plt.gca()
for i in [0, 1]:
    ax.plot(smooth(metrics[f"acc_{i}"]), alpha=0.8, linewidth=2)
ax.plot(smooth(metrics["ERM_acc_0"]), c="dimgray", alpha=0.5, linewidth=2)
ax.set_ylabel("Accuracy")
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.xaxis.set_tick_params(width=1.2)
ax.yaxis.set_tick_params(width=1.2)
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
savefig(f"temp/{exp_name}/learning_curve_with_ERM")

#%% Stitch figures into gifs
import imageio

os.makedirs("gifs", exist_ok=True)

filenames = [f"figures/temp/{exp_name}/{t}_h0.png" for t in fig_save_times]
images = [imageio.imread(filename) for filename in filenames]
gif_head_0_filename = f"gifs/{exp_name}_h0.gif"
imageio.mimsave(gif_head_0_filename, images)

filenames = [f"figures/temp/{exp_name}/{t}_h1.png" for t in fig_save_times]
images = [imageio.imread(filename) for filename in filenames]
gif_head_1_filename = f"gifs/{exp_name}_h1.gif"
imageio.mimsave(gif_head_1_filename, images)

filenames = [
    f"figures/temp/{exp_name}/learning_curve_full_{t}.png" for t in fig_save_times
]
images = [imageio.imread(filename) for filename in filenames]
gif_curve_filename = f"gifs/{exp_name}_curve.gif"
imageio.mimsave(gif_curve_filename, images)

print("GIF creation complete! Files are in:")
for fn in [gif_head_0_filename, gif_head_1_filename, gif_curve_filename]:
    print(fn)

# %%

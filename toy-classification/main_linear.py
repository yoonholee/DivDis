import argparse
import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from task_generator import generate_data, sample_minibatch
from utils import savefig

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from divdis import DivDisLoss

os.makedirs("figures/linear", exist_ok=True)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
training_data = generate_data(500, train=True)
test_data = generate_data(5000, train=False)

plt.figure(figsize=(4, 4))
tr_x, tr_y = training_data
te_x, _ = test_data

tr_g = tr_x[tr_y.flatten() == 0][:15]
class_1 = plt.scatter(
    tr_g[:, 0],
    tr_g[:, 1],
    marker="s",
    zorder=10,
    s=50,
    c="firebrick",
    edgecolors="k",
    linewidth=1,
)
tr_g = tr_x[tr_y.flatten() == 1][:15]
class_2 = plt.scatter(
    tr_g[:, 0],
    tr_g[:, 1],
    marker="^",
    zorder=10,
    s=70,
    c="royalblue",
    edgecolors="k",
    linewidth=1,
)

unlabeled = plt.scatter(
    te_x[:20, 0], te_x[:20, 1], zorder=0, s=30, c="silver", edgecolors="k", linewidth=1
)
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
savefig(f"linear/data")

legend_fig = plt.figure()
legend_fig.legend(
    [class_2, class_1, unlabeled],
    ["Class 1", "Class 2", "Unlabeled"],
    loc="center",
    ncol=3,
)
legend_fig.savefig("figures/linear_legend.pdf", bbox_inches="tight")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--test_batch_size", type=int, default=100)
parser.add_argument("--train_iter", type=int, default=20000)
parser.add_argument("--log_every", type=int, default=500)
parser.add_argument("--plot_every", type=int, default=1000)

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--heads", type=int, default=20)
parser.add_argument("--aux_weight", type=float, default=1.0)
parser.add_argument("--mode", type=str, default="mi")
parser.add_argument("--reduction", type=str, default="mean")

args = parser.parse_args()
args.noise_level = 0.0

exp_name = f"linear_h{args.heads}_{args.mode}-{args.reduction}w{args.aux_weight}"
print(exp_name)

net = nn.Linear(2, args.heads, bias=True).cuda()
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
loss_fn = DivDisLoss(heads=args.heads, mode=args.mode, reduction=args.reduction)


for t in range(args.train_iter + 1):
    x, y = sample_minibatch(training_data, args.batch_size)
    x, y = x.cuda(), y.cuda()
    logits = net(x)
    logits_chunked = torch.chunk(logits, args.heads, dim=-1)
    losses = [F.binary_cross_entropy_with_logits(logit, y) for logit in logits_chunked]
    xent = sum(losses)

    target_x, _ = sample_minibatch(test_data, args.test_batch_size)
    target_x = target_x.cuda()
    target_logits = net(target_x)
    repulsion_loss = loss_fn(target_logits)

    full_loss = xent + args.aux_weight * repulsion_loss
    opt.zero_grad()
    full_loss.backward()
    opt.step()

    if t % args.log_every == 0:
        print(f"{t=} xent {xent.item():.5f} aux {repulsion_loss.item():.5f}")

    times = sorted([2**n for n in range(15)] + [1000 * n for n in range(200)])
    times = [t for t in times if t < args.train_iter and t > 0]
    if t in times:
        plt.figure(figsize=(4, 4))

        weights = net.weight.detach().cpu()
        xs = np.arange(-1.05, 1.05, 0.01)
        plt.xlim([-1.05, 1.05])
        plt.ylim([-1.05, 1.05])

        def plot_linear_fn(xs, slope, intercept=0.0):
            ys = slope * xs + intercept
            plt.plot(xs, ys)

        for function_idx in range(len(weights)):
            w_0, w_1 = weights[function_idx][0].item(), weights[function_idx][1].item()
            slope = -w_0 / w_1
            _bias = net.bias[function_idx].detach().cpu().item()
            intercept = -_bias / w_1
            plot_linear_fn(xs, slope, intercept)

        tr_x, tr_y = training_data
        for g, c, m, s in [(0, "firebrick", "s", 50), (1, "royalblue", "^", 70)]:
            tr_g = tr_x[tr_y.flatten() == g][:15]
            plt.scatter(
                tr_g[:, 0],
                tr_g[:, 1],
                marker=m,
                zorder=10,
                s=s,
                c=c,
                edgecolors="k",
                linewidth=1,
            )
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        savefig(f"linear/{exp_name}_{t=}")

filenames = [f"figures/linear/{exp_name}_{t=}.png" for t in times]
images = [imageio.imread(filename) for filename in filenames]
os.makedirs("gifs", exist_ok=True)
imageio.mimsave(f"gifs/{exp_name}.gif", images)
print(f"Saved gif to gifs/{exp_name}.gif")

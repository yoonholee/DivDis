import os

import matplotlib.pyplot as plt


def savefig(name, transparent=False, pdf=False):
    FIG_ROOT = "figures"
    os.makedirs(FIG_ROOT, exist_ok=True)
    modes = ["png"]
    if pdf:
        modes += ["pdf"]
    for mode in modes:
        file_name = f"{FIG_ROOT}/{name}.{mode}"
        if transparent:
            plt.savefig(file_name, dpi=300, bbox_inches="tight", transparent=True)
        else:
            plt.savefig(file_name, dpi=300, facecolor="white", bbox_inches="tight")
    plt.show()
    plt.clf()

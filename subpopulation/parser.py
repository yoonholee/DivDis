import argparse
import os
from datetime import datetime

from data.data import dataset_attributes, shift_types
from models import model_attributes
from utils import ParseKwargs
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_args(args):
    if args.shift_type == "confounder":
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith("label_shift"):
        assert args.minority_fraction
        assert args.imbalance_ratio


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--setting", default="")

    parser.add_argument(
        "-d", "--dataset", choices=dataset_attributes.keys(), default="CMNIST"
    )
    parser.add_argument("-s", "--shift_type", choices=shift_types, default="confounder")

    # Confounders
    parser.add_argument("-t", "--target_name", default="waterbird_complete95")
    parser.add_argument(
        "-c", "--confounder_names", nargs="+", default=["forest2water2"]
    )
    # Resume?
    parser.add_argument("--resume", default=False, action="store_true")
    # Label shifts
    parser.add_argument("--minority_fraction", type=float)
    parser.add_argument("--imbalance_ratio", type=float)
    # Data
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--reweight_groups", action="store_true", default=False)
    parser.add_argument("--augment_data", action="store_true", default=False)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--dog_group", type=int, default=4)
    parser.add_argument("--truck_group", type=int, default=4)
    # Objective
    parser.add_argument("--robust", default=False, action="store_true")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--generalization_adjustment", default="0.0")
    parser.add_argument("--automatic_adjustment", default=False, action="store_true")
    parser.add_argument("--robust_step_size", default=0.01, type=float)
    parser.add_argument("--use_normalized_loss", default=False, action="store_true")
    parser.add_argument("--btl", default=False, action="store_true")
    parser.add_argument("--hinge", default=False, action="store_true")

    # Model
    parser.add_argument("--model", choices=model_attributes.keys(), default="resnet50")
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument(
        "--model_kwargs",
        nargs="*",
        action=ParseKwargs,
        default={},
        help="keyword arguments for model initialization passed as key1=value1 key2=value2",
    )

    # Diverse Ensemble
    parser.add_argument("--diversify", action="store_true", default=False)
    parser.add_argument("--head_only", action="store_true", default=False)
    parser.add_argument(
        "--majority_only",
        action="store_true",
        default=False,
        help="Use only majority classes during training.",
    )
    parser.add_argument("--majority_setting", type=str, default="")
    parser.add_argument(
        "--bn_mode", type=str, default="train", choices=["eval", "train", "mix"]
    )
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mode", type=str, default="mi")
    parser.add_argument("--reduction", type=str, default="mean")
    parser.add_argument("--diversity_weight", type=float, default=10.0)
    parser.add_argument("--reg_weight", type=float, default=10.0)
    parser.add_argument(
        "--reg_mode",
        type=str,
        default="kl_backward",
        choices=[
            "ratio",
            "entropy",
            "kl_forward",
            "kl_backward",
            "kl_ratio_f",
            "kl_ratio_b",
        ],
    )
    parser.add_argument("--fixed_label_ratio", type=float, default=None)
    parser.add_argument(
        "--ratio_split", type=str, default="source", choices=["source", "target"]
    )

    # Optimization
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--minimum_variational_weight", type=float, default=0)
    parser.add_argument("--num_warmup_steps", default=0, type=int)

    # Misc
    parser.add_argument("--in_dist_testing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", default="default")
    parser.add_argument("--log_every", default=50, type=int)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--save_best", action="store_true", default=False)
    parser.add_argument("--save_last", action="store_true", default=False)
    parser.add_argument(
        "--save_wrong",
        action="store_true",
        default=False,
        help="prepare for the second run of JTT",
    )
    parser.add_argument("--fold", default=None)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--wrong_file", type=str, default=None)
    parser.add_argument("--is_featurizer", type=int, default=True)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--step_gamma", type=float, default=0.96)
    parser.add_argument("--group_by_label", action="store_true", default=False)
    parser.add_argument("--exp_string", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    args.start_time = datetime.now()
    if args.setting == "CMNIST":
        args.shift_type = "confounder"
        args.dataset = "CMNIST"
        args.target_name = "0-4"
        args.confounder_names = ["isred"]
        args.lr = 0.001
        args.weight_decay = 0.0001
        args.batch_size = 16
        args.model = "resnet50"
        args.n_epochs = 100
        args.gamma = 0.1
    elif args.setting == "WBIRDS":
        args.shift_type = "confounder"
        args.dataset = "CUB"
        args.target_name = "waterbird_complete95"
        args.confounder_names = ["forest2water2"]
        args.lr = 0.001
        args.weight_decay = 0.0001
        args.model = "resnet50"
        args.n_epochs = 100
        args.gamma = 0.1
    elif args.setting == "MultiNLI":
        args.shift_type = "confounder"
        args.dataset = "MultiNLI"
        args.target_name = "gold_label_random"
        args.confounder_names = ["sentence2_has_negation"]
        args.lr = 2e-5
        args.weight_decay = 0.0
        args.model = "bert"
        args.n_epochs = 3
        args.reweight_groups = True
        args.robust = True
        args.batch_size = 16
    elif args.setting == "CELEBA_O":
        args.shift_type = "confounder"
        args.dataset = "CelebA"
        args.lr = 0.0001
        args.weight_decay = 0.0001
        args.model = "resnet50"
        args.n_epochs = 50
        args.batch_size = 32
        args.gamma = 0.1
        args.target_name = "Blond_Hair"
        args.confounder_names = ["Male"]
    elif "CELEBA" in args.setting:
        args.shift_type = "confounder"
        args.dataset = "CelebA"
        args.lr = 0.0001
        args.weight_decay = 0.0001
        args.model = "resnet50"
        args.n_epochs = 10
        args.batch_size = 32
        args.gamma = 0.1
        if args.setting == "CELEBA":
            args.target_name = "Blond_Hair"
            args.confounder_names = ["Male"]
        elif args.setting == "CELEBA_1":
            args.target_name = "Mouth_Slightly_Open"
            args.confounder_names = ["Wearing_Lipstick"]
        elif args.setting == "CELEBA_2":
            args.target_name = "Attractive"
            args.confounder_names = ["Smiling"]
        elif args.setting == "CELEBA_3":
            args.target_name = "Wavy_Hair"
            args.confounder_names = ["High_Cheekbones"]
        elif args.setting == "CELEBA_4":
            args.target_name = "Heavy_Makeup"
            args.confounder_names = ["Big_Lips"]

    if args.debug:
        args.exp_string += "__DEBUG__"
        args.log_dir = "__DEBUG__"
        args.n_epochs = 2

    args.log_dir = os.path.join("./logs", args.log_dir)
    check_args(args)
    if "CELEBA" in args.setting:
        args.exp_string += args.setting
    else:
        args.exp_string += args.dataset

    if args.augment_data:
        args.exp_string += "_aug"
    if args.diversify:
        args.exp_string += f"_div_h{args.heads}_{args.mode}-{args.reduction}-{args.diversity_weight:.2f}_{args.reg_mode}-{args.reg_weight:.2f}"
        if args.bn_mode != "train":
            args.exp_string += f"_{args.bn_mode}"
        if args.majority_only:
            args.exp_string += "_maj"
            if len(args.majority_setting) > 0:
                args.exp_string += args.majority_setting
        if args.batch_size != 16:
            args.exp_string += f"_bs{args.batch_size}"
        if args.lr != 1e-3:
            args.exp_string += f"_lr{args.lr}"
        if args.fixed_label_ratio:
            args.exp_string += f"_fl{args.fixed_label_ratio}"
        elif args.ratio_split == "target":
            args.exp_string += "_tr"
    if args.in_dist_testing:
        args.exp_string += "_idtest"
    if args.robust:
        args.exp_string += "_robust"
    if args.reweight_groups:
        args.exp_string += "_reweight"
    if args.dataset == "MetaDatasetCatDog":
        args.exp_string += f"_dog_{int(args.dog_group)}"
    if args.weight_decay >= 0.01:
        args.exp_string += f"_penalty_{args.weight_decay}"
    args.exp_string += f"_{args.seed}"

    # BERT-specific configs copied over from run_glue.py
    if "bert" in args.model:
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0
    return args

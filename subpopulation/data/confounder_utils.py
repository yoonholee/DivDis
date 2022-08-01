import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from models import model_attributes
from PIL import Image
from torch.utils.data import Dataset

from data.celebA_dataset import CelebADataset
from data.cmnist_dataset import CMNISTDataset
from data.cub_dataset import CUBDataset
from data.dro_dataset import DRODataset
from data.meta_dataset_cat_dog import MetaDatasetCatDog
from data.multinli_dataset import MultiNLIDataset

################
### SETTINGS ###
################

confounder_settings = {
    "CelebA": {"constructor": CelebADataset},
    "CUB": {"constructor": CUBDataset},
    "MetaDatasetCatDog": {"constructor": MetaDatasetCatDog},
    "CMNIST": {"constructor": CMNISTDataset},
    "MultiNLI": {"constructor": MultiNLIDataset},
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]["constructor"](
        args=args,
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
    )
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
    if train:
        splits = ["train", "val", "test"]
    else:
        splits = ["test"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        for split in splits
    ]
    return dro_subsets


def prepare_group_confounder_data(
    args, group_id, train_data=None, return_full_dataset=False
):

    full_dataset = confounder_settings[args.dataset]["constructor"](
        args=args,
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        group_id=group_id,
    )
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )

    splits = ["train"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
        for split in splits
    ]
    return dro_subsets[0]

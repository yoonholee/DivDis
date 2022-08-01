import os
import pdb

import torch
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision import datasets
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate(
            [np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)],
            axis=2,
        )
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
      root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
      env (string): Which environment to load. Must be 1 of 'train1', 'val', 'test', or 'all_train'.
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    def __init__(
        self, root="./data", env="train1", transform=None, target_transform=None
    ):
        super(ColoredMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.prepare_colored_mnist()
        if env in ["train1", "val", "test"]:
            self.data_label_tuples = torch.load(
                os.path.join(self.root, "ColoredMNIST", env) + ".pt"
            )
        elif env == "all_train":
            self.data_label_tuples = torch.load(
                os.path.join(self.root, "ColoredMNIST", "train1.pt")
            ) + torch.load(os.path.join(self.root, "ColoredMNIST", "val.pt"))
        else:
            raise RuntimeError(
                f"{env} env unknown. Valid envs are train1, val, test, and all_train"
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, c = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, c

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, "ColoredMNIST")
        if (
            os.path.exists(os.path.join(colored_mnist_dir, "train1.pt"))
            and os.path.exists(os.path.join(colored_mnist_dir, "val.pt"))
            and os.path.exists(os.path.join(colored_mnist_dir, "test.pt"))
        ):
            print("Colored MNIST dataset already exists")
            return

        print("Preparing Colored MNIST")
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        val_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f"Converting image {idx}/{len(train_mnist)}")
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 30000:
                # 20% in the first training environment
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.5:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 30000:
                train1_set.append(
                    (Image.fromarray(colored_arr), binary_label, color_red)
                )
            elif idx < 40000:
                val_set.append((Image.fromarray(colored_arr), binary_label, color_red))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label, color_red))

            # Debug
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
            # break

        #     dataset_utils.makedir_exist_ok(colored_mnist_dir)
        os.makedirs(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, "train1.pt"))
        torch.save(val_set, os.path.join(colored_mnist_dir, "val.pt"))
        torch.save(test_set, os.path.join(colored_mnist_dir, "test.pt"))


class CMNISTDataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(
        self,
        args,
        root_dir,
        target_name,
        confounder_names,
        model_type,
        augment_data,
        mix_up=False,
        mix_alpha=2,
        mix_unit="group",
        mix_type=1,
        mix_freq="batch",
        group_id=None,
        dataset=None,
    ):
        self.args = args
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type
        self.mix_up = mix_up
        self.mix_alpha = mix_alpha
        self.mix_unit = mix_unit
        self.mix_type = mix_type
        self.group_id = group_id
        self.RGB = True
        # Read in attributes
        # self.attrs_df = pd.read_csv(
        #     os.path.join(root_dir, 'data', 'list_attr_celeba.csv'))

        self.colored_mnist_train = ColoredMNIST(root=root_dir, env="train1")
        self.colored_mnist_val = ColoredMNIST(root=root_dir, env="val")
        self.colored_mnist_test = ColoredMNIST(root=root_dir, env="test")
        self.precomputed = True
        self.pretransformed = False
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.0), (0.3081, 0.3081, 0.3081)),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.0), (0.3081, 0.3081, 0.3081)),
            ]
        )
        self.n_classes = 2
        self.n_confounders = 1

        self.color_array = np.array(
            [x[2] for x in self.colored_mnist_train.data_label_tuples]
            + [x[2] for x in self.colored_mnist_val.data_label_tuples]
            + [x[2] for x in self.colored_mnist_test.data_label_tuples]
        )

        self.features_mat = (
            [x[0] for x in self.colored_mnist_train.data_label_tuples]
            + [x[0] for x in self.colored_mnist_val.data_label_tuples]
            + [x[0] for x in self.colored_mnist_test.data_label_tuples]
        )
        self.y_array = np.array(
            [x[1] for x in self.colored_mnist_train.data_label_tuples]
            + [x[1] for x in self.colored_mnist_val.data_label_tuples]
            + [x[1] for x in self.colored_mnist_test.data_label_tuples]
        )
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(
            1, torch.tensor(self.y_array, dtype=torch.int64).unsqueeze(1), 1
        ).numpy()

        self.n_groups = 4
        self.group_array = 2 * self.color_array + self.y_array
        self.split_dict = {"train": 0, "val": 1, "test": 2}
        self.train_split_array = np.zeros(30000)
        self.val_split_array = np.ones(10000) * 1
        self.test_split_array = np.ones(20000) * 2

        self.split_array = np.concatenate(
            [self.train_split_array, self.val_split_array, self.test_split_array]
        )
        self.mix_array = [False] * len(self.y_array)

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.features_mat = self.features_mat[idxes]
            self.group_array = self.group_array[idxes]
            self.split_array = self.split_array[idxes]
            self.y_array = self.y_array[idxes]
            self.y_array_onehot = self.y_array_onehot[idxes]

        if args.group_by_label:
            idxes = np.where(self.split_array == self.split_dict["train"])[0]
            self.group_array[idxes] = self.y_array[idxes]

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def augment_with_cross_mix_up(self, train_g, train_y_onehot):

        augmented_y = []
        augmented_group_id = []
        augmented_mix_weight = []
        augmented_mix_idx = []

        unique_group_id, counts = np.unique(train_g, return_counts=True)
        max_group_num = np.max(counts)
        if self.mix_type == 1:
            y1, group_id1, weight1, mix_idx1 = self.cross_mix_up(
                train_g, train_y_onehot, max_group_num, 0, 1
            )
            y2, group_id2, weight2, mix_idx2 = self.cross_mix_up(
                train_g, train_y_onehot, max_group_num, 2, 3
            )
        else:
            y1, group_id1, weight1, mix_idx1 = self.cross_mix_up(
                train_g, train_y_onehot, max_group_num, 0, 2
            )
            y2, group_id2, weight2, mix_idx2 = self.cross_mix_up(
                train_g, train_y_onehot, max_group_num, 1, 3
            )

        augmented_y.extend(y1 + y2)
        augmented_group_id.extend(group_id1 + group_id2)
        augmented_mix_weight.extend(weight1 + weight2)
        augmented_mix_idx.extend(mix_idx1 + mix_idx2)

        return augmented_y, augmented_group_id, augmented_mix_weight, augmented_mix_idx

    def cross_mix_up(
        self, train_g, train_y_onehot, max_group_num, group_id_1, group_id_2
    ):

        augmented_y, augmented_group_id, augmented_mix_weight, augmented_mix_idx = (
            [],
            [],
            [],
            [],
        )

        idxes1, idxes2 = (
            np.where(train_g == group_id_1)[0],
            np.where(train_g == group_id_2)[0],
        )

        num = 2 * max_group_num - len(idxes1) - len(idxes2)
        l = np.random.beta(self.mix_alpha, self.mix_alpha) * np.ones([num, 1])
        l_y = np.tile(l, [1, self.n_classes])
        l_y = torch.tensor(l_y, dtype=torch.float32)

        selected_idxes_1 = np.random.choice(idxes1, size=num)
        selected_idxes_2 = np.random.choice(idxes2, size=num)

        mixed_y = (
            l_y * train_y_onehot[selected_idxes_1]
            + (1 - l_y) * train_y_onehot[selected_idxes_2]
        )

        augmented_y.append(mixed_y)
        augmented_group_id.append([4] * len(mixed_y))
        augmented_mix_idx.append(
            np.concatenate(
                [selected_idxes_1.reshape(-1, 1), selected_idxes_2.reshape(-1, 1)],
                axis=1,
            )
        )
        augmented_mix_weight.append(l.reshape(-1))

        augmented_y = np.concatenate(augmented_y, axis=0)
        augmented_group_id = np.array(augmented_group_id)
        augmented_mix_weight = np.concatenate(augmented_mix_weight, axis=0)
        augmented_mix_idx = np.concatenate(augmented_mix_idx, axis=0)

        return augmented_y, augmented_group_id, augmented_mix_weight, augmented_mix_idx

    def augment_with_mix_up(self, train_g, train_y_onehot):
        augmented_y = []
        augmented_mix_idx = []
        augmented_mix_weight = []
        augmented_group_id = []

        unique_group_id, counts = np.unique(train_g, return_counts=True)
        max_group_num = np.max(counts)
        for group_id, count in zip(unique_group_id, counts):
            idxes = np.where(train_g == group_id)[0]

            if count == max_group_num:
                continue

            l = np.random.beta(self.mix_alpha, self.mix_alpha) * np.ones(
                [max_group_num - count, 1]
            )

            l_y = np.tile(l, [1, self.n_classes])
            l_y = torch.tensor(l_y, dtype=torch.float32)

            idxes_1 = np.random.choice(idxes, size=max_group_num - count)
            idxes_2 = np.random.choice(idxes, size=max_group_num - count)

            mixed_y = (
                l_y * train_y_onehot[idxes_1] + (1 - l_y) * train_y_onehot[idxes_2]
            )

            augmented_y.append(mixed_y)
            augmented_group_id.extend([group_id] * len(mixed_y))
            augmented_mix_idx.append(
                np.concatenate([idxes_1.reshape(-1, 1), idxes_2.reshape(-1, 1)], axis=1)
            )
            augmented_mix_weight.append(l.reshape(-1))

        augmented_y = np.concatenate(augmented_y, axis=0)
        augmented_group_id = np.array(augmented_group_id)
        augmented_mix_weight = np.concatenate(augmented_mix_weight, axis=0)
        augmented_mix_idx = np.concatenate(augmented_mix_idx, axis=0)

        return augmented_y, augmented_group_id, augmented_mix_weight, augmented_mix_idx

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

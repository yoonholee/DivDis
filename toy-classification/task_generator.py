import numpy as np
import torch


def generate_data(num_datapoints, train=True, swap_y_meaning=False, noise_level=None):
    # x1 \in [-2, 0] and y is true if x1<-1
    # x2 \in [-1, 1] and x2 is >0 iff x1<-1 (if train, else they are uncorrelated)

    x1 = np.random.rand(num_datapoints, 1) * 2 - 1
    y = (x1 < 0).astype(int)
    if train == False:
        # Inputs are uncorrelated
        x2 = np.random.rand(num_datapoints, 1) * 2 - 1
    else:
        # Generate inputs in both cases and then combine in correlated way
        x2a = np.random.rand(num_datapoints, 1)
        x2b = np.random.rand(num_datapoints, 1) - 1
        x2 = (x1 < 0).astype(float) * x2a + (x1 >= 0).astype(float) * x2b

    if noise_level:
        x1 += (np.random.rand(num_datapoints, 1) - 0.5) * noise_level

    x = np.concatenate([x1, x2], 1)
    if swap_y_meaning:
        # for the purpose of generating an alternative test set.
        y = (x2 > 0).astype(int)
    return x, y


def generate_data_ndim(num_datapoints, dims=2, train=True):
    y = np.random.binomial(1, 0.5, num_datapoints).reshape(-1, 1)
    rand_zeroone = np.random.rand(num_datapoints, dims)
    y_to_sign = 2 * y - 1
    if train:
        x = rand_zeroone * y_to_sign
    else:
        x = np.zeros_like(rand_zeroone)
        x[:, 0] = rand_zeroone[:, 0] * y_to_sign.squeeze()
        x[:, 1:] = rand_zeroone[:, 1:] * 2 - 1
    return x, y


def sample_minibatch(data, batch_size):
    x, y = data
    minibatch_idx = np.random.randint(0, x.shape[0], size=batch_size)
    return (
        torch.tensor(x[minibatch_idx]).float(),
        torch.tensor(y[minibatch_idx]).float(),
    )

"""
Example of using the MNIST dataset with the optimal transport loss function.
"""

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np


from sinkhorn.opt_transport import entropy_regularized_opt_transport_dual_dist
from utils.const import DATA_DIR


def load_mnist() -> tuple[datasets.MNIST, datasets.MNIST]:
    """
    Load the MNIST dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    return mnist_train, mnist_test


class ImageToDistribution:
    """
    Convert an image to a distribution.
    """

    def __call__(self, tensor):
        low = tensor.min()
        return ((tensor - low) / tensor.sum()).flatten().squeeze()


def get_mnist_loaders(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 1,
    transform=None,
):
    """
    Returns DataLoader objects for MNIST train and test datasets, with optional custom transform.

    Args:
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the training data.
        num_workers (int): Number of subprocesses to use for data loading.
        custom_transform (callable, optional): Custom transform to apply to the images.

    Returns:
        train_loader, test_loader: DataLoader objects for train and test sets.
    """
    # If custom_transform is not None, assume it will handle normalization.
    # Otherwise, define a transform that normalizes each image by its own mean and std.
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                ImageToDistribution(),
            ]
        )

    mnist_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    mnist_test = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


class SinkhornLoss(nn.Module):
    """
    Optimal transport based loss function.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self):
        """
        Forward pass.
        """
        raise NotImplementedError


def optimal_transport_cost_mat(img_dim: int):
    """
    Compute the optimal transport cost matrix.
    """
    cost = np.zeros((img_dim**2, img_dim**2))
    for i in range(img_dim**2):
        x = i % img_dim
        y = i // img_dim
        for j in range(img_dim**2):
            x2 = j % img_dim
            y2 = j // img_dim
            cost[i, j] = np.linalg.norm((x - x2, y - y2))
    return cost


if __name__ == "__main__":
    train, test = get_mnist_loaders()
    COST_MAT = None

    for i, [img, label] in enumerate(train):
        img1, img2 = img[0], img[1]

        if COST_MAT is None:
            COST_MAT = optimal_transport_cost_mat(int(np.sqrt(img1.shape[0])))

        entropy_reg_dist = entropy_regularized_opt_transport_dual_dist(
            img1.numpy(), img2.numpy(), COST_MAT, 9
        )
        print("--------------------------------")
        print(entropy_reg_dist)
        print("label 1:", label[0], "label 2:", label[1])
        entropy_reg_dist = entropy_regularized_opt_transport_dual_dist(
            img1.numpy(), img1.numpy(), COST_MAT, 9
        )
        print("--------------------------------")
        print(entropy_reg_dist)
        print("label 1:", label[0], "label 2:", label[0])

        if i >= 5:
            break

"""
Example of using the MNIST dataset with the optimal transport loss function.
"""

from torchvision import datasets, transforms
import torch
from torch import nn
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


class MNISTClassifier:
    """
    MNIST classifier.
    """

    def __init__(
        self,
        train_dataset: datasets.MNIST,
        num_imgs_per_label: int = 10,
        lmbda: float = 9,
    ):
        train_imgs = {}
        num_filled = 0
        img_dim = None

        for img, label in train_dataset:
            img = img[0, :]
            label = label[0]
            if img_dim is None:
                img_dim = int(np.sqrt(img.shape[0]))

            img_list = train_imgs.get(label, [])
            if len(img_list) >= num_imgs_per_label:
                continue

            img_list.append(img)
            if len(img_list) >= num_imgs_per_label:
                num_filled += 1

            train_imgs[label] = img_list
            if num_filled >= num_imgs_per_label:
                break

        self.train_imgs = train_imgs
        self.cost_mat = optimal_transport_cost_mat(img_dim)
        self.lmbda = lmbda

    def classify(self, img: torch.Tensor) -> int:
        """
        Classify an image.
        """
        img = img[0, :]

        opt_dist = {
            label: np.mean(
                [
                    entropy_regularized_opt_transport_dual_dist(
                        img.numpy(), train_img.numpy(), self.cost_mat, self.lmbda
                    )
                    for train_img in train_imgs
                ]
            )
            for label, train_imgs in self.train_imgs.items()
        }
        print(f"opt_dist: {opt_dist}")

        return min(opt_dist, key=opt_dist.get)


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
    cost = np.zeros((img_dim**2, img_dim**2))
    for i in range(img_dim**2):
        x = i % img_dim
        y = i // img_dim
        for j in range(img_dim**2):
            x2 = j % img_dim
            y2 = j // img_dim
            cost[i, j] = np.linalg.norm((x - x2, y - y2))
    return cost

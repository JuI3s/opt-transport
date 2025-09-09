from torchvision import datasets, transforms
import torch
import torch.nn as nn


from utils.const import DATA_DIR


def load_mnist() -> tuple[datasets.MNIST, datasets.MNIST]:
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


class PerImageNormalize(object):
    def __call__(self, tensor):
        # tensor shape: [C, H, W]
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            std = 1.0
        return ((tensor - mean) / std).flatten().squeeze()


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
                PerImageNormalize(),
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

    def forward(self, input, target):
        raise NotImplementedError


if __name__ == "__main__":
    mnist_train, mnist_test = load_mnist()
    print(mnist_train.data.shape)
    print(mnist_test.data.shape)

    train_loader, test_loader = get_mnist_loaders()

    for i, [img, label] in enumerate(train_loader):
        print(img[0].shape)
        if i >= 0:
            break

"""
MNIST test
"""

import pytest

from example.mnist import MNISTClassifier, get_mnist_loaders


@pytest.mark.ignore
def test_mnist():
    """
    Test the MNIST dataset classifier.
    """

    train, test = get_mnist_loaders()
    classifier = MNISTClassifier(train)
    num_samples = 5
    num_right = 0
    for i, (img, label) in enumerate(test):
        pred = classifier.classify(img)
        print(f"Predicted label: {pred}, Actual label: {label[0]}")
        if pred == label[0]:
            num_right += 1
        if i + 1 >= num_samples:
            break
    print(f"right: {num_right}/{num_samples}")

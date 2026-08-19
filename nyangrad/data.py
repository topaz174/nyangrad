"""Datasets, image transforms, and minibatch iteration."""

import gzip
import struct
from collections.abc import Callable, Sequence

import numpy as np

from .autograd import Tensor


class Dataset:
    """A collection that supports integer indexing and len()."""

    def __init__(self, transforms: Sequence[Callable] | None = None):
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, value):
        if self.transforms:
            for transform in self.transforms:
                value = transform(value)
        return value


class DataLoader:
    """Iterate over a dataset as Tensor batches."""

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ordering: list[np.ndarray] = []
        self.index = 0

    def __iter__(self):
        indices = (
            np.random.permutation(len(self.dataset))
            if self.shuffle
            else np.arange(len(self.dataset))
        )
        self.ordering = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.ordering):
            raise StopIteration

        batch_indices = self.ordering[self.index]
        self.index += 1
        samples = [self.dataset[int(index)] for index in batch_indices]
        columns = zip(*samples)
        return tuple(Tensor(np.asarray(column)) for column in columns)


class Transform:
    def __call__(self, value):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    """Flip an HWC image from left to right with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image):
        return np.flip(image, axis=1) if np.random.rand() < self.p else image


class RandomCrop(Transform):
    """Translate an HWC image within zero padding, keeping its original size."""

    def __init__(self, padding: int = 3):
        self.padding = padding

    def __call__(self, image):
        shift_y, shift_x = np.random.randint(
            -self.padding, self.padding + 1, size=2
        )
        height, width = image.shape[:2]
        padded = np.pad(
            image,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
        )
        top = self.padding + shift_y
        left = self.padding + shift_x
        return padded[top : top + height, left : left + width]


class MNISTDataset(Dataset):
    """MNIST images and labels read directly from the gzipped IDX files."""

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Sequence[Callable] | None = None,
    ):
        super().__init__(transforms)

        with gzip.open(image_filename, "rb") as images_file:
            magic, count, rows, cols = struct.unpack(">IIII", images_file.read(16))
            if magic != 2051:
                raise ValueError(f"invalid MNIST image magic number: {magic}")
            raw = images_file.read(count * rows * cols)
            self.images = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(count, rows, cols, 1)
                .astype(np.float32)
                / 255.0
            )

        with gzip.open(label_filename, "rb") as labels_file:
            magic, label_count = struct.unpack(">II", labels_file.read(8))
            if magic != 2049:
                raise ValueError(f"invalid MNIST label magic number: {magic}")
            self.labels = np.frombuffer(
                labels_file.read(label_count), dtype=np.uint8
            )

        if len(self.images) != len(self.labels):
            raise ValueError("MNIST image and label counts do not match")

    def __getitem__(self, index):
        return self.apply_transforms(self.images[index]), self.labels[index]

    def __len__(self) -> int:
        return len(self.images)


class NDArrayDataset(Dataset):
    """Expose arrays with the same leading dimension as one dataset."""

    def __init__(self, *arrays):
        if not arrays:
            raise ValueError("at least one array is required")
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError("all arrays must have the same leading dimension")
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

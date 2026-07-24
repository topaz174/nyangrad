import gzip
import struct

import numpy as np

from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.index = 0

        if self.shuffle:
            self.ordering = np.array_split(np.random.permutation(len(self.dataset)), 
                                           range(self.batch_size, len(self.dataset), self.batch_size))

        self.total_batches = len(self.ordering)

        return self

    def __next__(self):
        if self.index >= self.total_batches:
            raise StopIteration

        curr_batch_indices = self.ordering[self.index]        
        self.index += 1

        images = []
        labels = []

        for idx in curr_batch_indices:
            images.append(self.dataset[idx][0])
            if len(self.dataset[idx]) > 1:
                labels.append(self.dataset[idx][1])
            
        images_T = Tensor(np.array(images))
        labels_T = Tensor(np.array(labels))

        return images_T, labels_T



class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, axis=1)
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        img_padded = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant', constant_values=0)

        x_end = -(self.padding - shift_x)
        x_end = x_end if x_end != 0 else None
        y_end = -(self.padding - shift_y)
        y_end = y_end if y_end != 0 else None

        return img_padded[self.padding + shift_x : x_end, self.padding + shift_y : y_end]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)

        with gzip.open(image_filename, 'rb') as f:
            header = f.read(16)
            magic, nums, rows, cols = struct.unpack('>IIII', header)

            data = f.read(nums * rows * cols)
            X_uint8 = np.frombuffer(data, dtype=np.uint8).reshape(nums, rows, cols, 1)
            X = X_uint8.astype(np.float32)
            self.images = X / 255
            
        with gzip.open(label_filename, 'rb') as f:
            header = f.read(8)
            magic, nums = struct.unpack('>II', header)

            data = f.read(nums)
            self.labels = np.frombuffer(data, dtype=np.uint8)

    def __getitem__(self, index) -> object:
        image = self.images[index]
        label = self.labels[index]

        image = self.apply_transforms(image)

        return image, label

    def __len__(self) -> int:
        return self.images.shape[0]

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
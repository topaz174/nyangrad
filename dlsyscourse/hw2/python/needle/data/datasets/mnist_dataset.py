from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]

        image = self.apply_transforms(image)

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
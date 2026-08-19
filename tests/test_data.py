import gzip
import struct

import numpy as np
import pytest

import nyangrad as nyan


class CountingDataset(nyan.data.Dataset):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.calls = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.calls += 1
        return np.array([index, index + 1]), index % 2


def test_dataset_base_methods_are_abstract_by_convention():
    dataset = nyan.data.Dataset()
    with pytest.raises(NotImplementedError):
        len(dataset)
    with pytest.raises(NotImplementedError):
        dataset[0]


def test_ndarray_dataset_indexes_arrays_together():
    features = np.arange(12).reshape(4, 3)
    labels = np.array([1, 0, 1, 0])
    dataset = nyan.data.NDArrayDataset(features, labels)

    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset[2][0], features[2])
    assert dataset[2][1] == labels[2]


def test_ndarray_dataset_rejects_bad_inputs():
    with pytest.raises(ValueError):
        nyan.data.NDArrayDataset()
    with pytest.raises(ValueError):
        nyan.data.NDArrayDataset(np.zeros((2, 1)), np.zeros((3, 1)))


def test_dataloader_batches_in_order_and_fetches_each_sample_once():
    dataset = CountingDataset(5)
    batches = list(nyan.data.DataLoader(dataset, batch_size=2))

    assert [batch[0].shape[0] for batch in batches] == [2, 2, 1]
    np.testing.assert_array_equal(batches[0][0].numpy(), [[0, 1], [1, 2]])
    np.testing.assert_array_equal(batches[-1][1].numpy(), [0])
    assert dataset.calls == len(dataset)


def test_dataloader_shuffle_changes_order_without_losing_samples():
    dataset = CountingDataset(20)
    loader = nyan.data.DataLoader(dataset, batch_size=6, shuffle=True)

    np.random.seed(0)
    first = np.concatenate([labels.numpy() for _, labels in loader])
    np.random.seed(1)
    second = np.concatenate([labels.numpy() for _, labels in loader])

    expected = [0] * 10 + [1] * 10
    assert sorted(first.tolist()) == sorted(second.tolist()) == expected
    assert not np.array_equal(first, second)


def test_dataloader_rejects_nonpositive_batch_size():
    with pytest.raises(ValueError):
        nyan.data.DataLoader(CountingDataset(1), batch_size=0)


def test_horizontal_flip_probability_endpoints():
    image = np.arange(12).reshape(2, 3, 2)
    np.testing.assert_array_equal(nyan.data.RandomFlipHorizontal(0)(image), image)
    np.testing.assert_array_equal(
        nyan.data.RandomFlipHorizontal(1)(image), np.flip(image, axis=1)
    )


def test_random_crop_keeps_shape_and_uses_zero_padding():
    image = np.ones((4, 5, 1), dtype=np.float32)

    np.random.seed(4)
    cropped = nyan.data.RandomCrop(padding=2)(image)

    assert cropped.shape == image.shape
    assert set(np.unique(cropped)) <= {0.0, 1.0}
    np.testing.assert_array_equal(nyan.data.RandomCrop(padding=0)(image), image)


def _write_idx_files(tmp_path, images, labels):
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"

    with gzip.open(image_path, "wb") as stream:
        stream.write(struct.pack(">IIII", 2051, *images.shape))
        stream.write(images.tobytes())
    with gzip.open(label_path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, len(labels)))
        stream.write(labels.tobytes())
    return image_path, label_path


def test_mnist_dataset_reads_normalizes_and_transforms(tmp_path):
    images = np.array(
        [
            [[0, 64], [128, 255]],
            [[255, 128], [64, 0]],
        ],
        dtype=np.uint8,
    )
    labels = np.array([3, 7], dtype=np.uint8)
    image_path, label_path = _write_idx_files(tmp_path, images, labels)
    dataset = nyan.data.MNISTDataset(
        image_path,
        label_path,
        transforms=[lambda image: image + 1],
    )

    assert len(dataset) == 2
    image, label = dataset[0]
    assert image.shape == (2, 2, 1)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[..., 0], images[0] / 255.0 + 1)
    assert label == 3


def test_mnist_dataset_checks_magic_numbers(tmp_path):
    images = tmp_path / "bad-images.gz"
    labels = tmp_path / "labels.gz"
    with gzip.open(images, "wb") as stream:
        stream.write(struct.pack(">IIII", 0, 1, 1, 1))
        stream.write(bytes([0]))
    with gzip.open(labels, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 1))
        stream.write(bytes([0]))

    with pytest.raises(ValueError, match="magic"):
        nyan.data.MNISTDataset(images, labels)

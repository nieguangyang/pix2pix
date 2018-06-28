import os
import numpy as np
from glob import glob
from scipy.misc import imread, imresize
from matplotlib import pyplot as plt
from img import IMG_DIR


DATA_DIR = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1])


def load_img(dataset, subset, rows=256):
    """
    Load images.
    :param dataset: str, name of dataset, such as "facade"
    :param subset: str, name of subset, normally one of "train", "val" and "test"
    :param rows: int, rows and columns of image
    :return real: ndarray, (n, rows, rows, 3) ndarray of n real images
    :return cond: ndarray, (n, rows, rows, 3) ndarray of n conditional images
    """
    npz_file = "%s/%s_%s_%d.npz" % (DATA_DIR, dataset, subset, rows)
    if os.path.exists(npz_file):
        print("Load img from npz file %s" % npz_file)
        data = np.load(npz_file)
        real = data["real"]
        cond = data["cond"]
        return real, cond
    dataset_subset_dir = "%s/%s/%s" % (IMG_DIR, dataset, subset)
    if not os.path.exists(dataset_subset_dir):
        raise Exception("No directory %s." % dataset_subset_dir)
    print("Load img from directory %s." % dataset_subset_dir)
    paths = glob("%s/*.jpg" % dataset_subset_dir)
    reals = []
    conds = []
    for path in paths:
        img = np.array(imresize(imread(path, mode="RGB"), (rows, 2 * rows))).reshape((1, rows, 2 * rows, 3))
        real = img[:, :, :rows, :]
        cond = img[:, :, rows:, :]
        reals.append(real)
        conds.append(cond)
    real = np.concatenate(reals, 0)
    cond = np.concatenate(conds, 0)
    np.savez(npz_file, real=real, cond=cond)
    return real, cond


cache = {}


def load_batch(dataset, subset, rows=256, batch_size=1):
    """
    Load images and yield batches.
    :param dataset: str, name of dataset, such as "facade"
    :param subset: str, name of subset, normally one of "train", "val" and "test"
    :param rows: int, rows and columns of image
    :param batch_size: int, batch size
    :yield real: ndarray, (batch_size, rows, rows, 3) ndarray of normalized real images of given batch size
    :yield cond: ndarray, (batch_size, rows, rows, 3) ndarray of normalized conditional images of given batch size
    """
    dataset_subset_rows = "%s_%s_%d" % (dataset, subset, rows)
    img = cache.get(dataset_subset_rows)
    if img:
        print("Load batch from cache.")
        _real, _cond = img
    else:
        _real, _cond = load_img(dataset, subset, rows)
        cache[dataset_subset_rows] = (_real, _cond)
    n = _real.shape[0]
    i = 0
    while i + batch_size < n:
        real = _real[i: i + batch_size] / 127.5 - 1
        cond = _cond[i: i + batch_size] / 127.5 - 1
        yield real, cond
        i += batch_size
    real = _real[i:] / 127.5 - 1
    cond = _cond[i:] / 127.5 - 1
    yield real, cond


def load_img_test():
    dataset = "facade"
    subset = "train"
    rows = 256
    real, cond = load_img(dataset, subset, rows)
    n = real.shape[0]
    for i in range(n):
        plt.subplot(121).imshow(real[i])
        plt.subplot(122).imshow(cond[i])
        plt.show()


def load_batch_test():
    dataset = "facade"
    subset = "train"
    rows = 256
    for real, cond in load_batch(dataset, subset, rows, 120):
        print(real.shape, cond.shape)
    for real, cond in load_batch(dataset, subset, rows, 1):
        _real = ((real + 1) * 127.5).astype(np.uint8)
        _cond = ((cond + 1) * 127.5).astype(np.uint8)
        plt.subplot(121).imshow(_real[0])
        plt.subplot(122).imshow(_cond[0])
        plt.show()


if __name__ == "__main__":
    # load_img_test()
    load_batch_test()

from argparse import ArgumentParser
import imageio
import numpy as np
import os
from scipy.ndimage.interpolation import zoom


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--data_root', type=str, default='data')
    a.add_argument('--selected_dataset', type=str, default='lfw_5590')
    a.add_argument('--out_dir', type=str, default='data/processed')
    return a.parse_args()


def run(flags):
    f = os.path.join(flags.data_root, 'training.txt')
    lines = open(f).readlines()
    lines = filter(lambda s: flags.selected_dataset in s, lines)
    lines = map(lambda s: s.strip(), lines)
    lines = sorted(lines)
    n = len(lines)

    filenames = [None] * n
    genders = np.zeros(n, 'uint8')
    smiles = np.zeros(n, 'uint8')
    glasses = np.zeros(n, 'uint8')
    poses = np.zeros(n, 'uint8')
    landmarks = np.zeros((n, 10), 'float32')
    for i, line in enumerate(lines):
        ss = line.split()
        filenames[i] = os.path.join(
            flags.data_root, ss[0].replace('\\', os.path.sep))
        genders[i], smiles[i], glasses[i], poses[i] = map(int, ss[-4:])
        landmarks[i, np.arange(10)] = tuple(map(float, ss[1:-4]))
    print()
    print(filenames[0])
    print(genders[0], smiles[0], glasses[0], poses[0])
    print(landmarks[0])
    print()
    filenames_set = set(filenames)

    images_dir = '%s/%s/' % (flags.data_root, flags.selected_dataset)
    ff = os.listdir(images_dir)
    ff = map(lambda f: os.path.join(images_dir, f), ff)
    ff = filter(lambda f: f in filenames_set, ff)
    ff = sorted(ff)
    assert filenames == ff

    f = ff[0]
    print(f)
    im = imageio.imread(f)
    x = np.array(im)
    x = x.transpose([2, 0, 1])
    print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())

    x = zoom(x, (1, 64 / x.shape[1], 64 / x.shape[2]))
    print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())


if __name__ == '__main__':
    run(parse_flags())

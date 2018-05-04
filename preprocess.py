from argparse import ArgumentParser
import imageio
import numpy as np
import os
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--data_root', type=str, default='data')
    a.add_argument('--selected_dataset', type=str, default='lfw_5590')
    a.add_argument('--out_dir', type=str, default='data/proc')
    return a.parse_args()


def process_labels(data_root, selected_dataset, out_dir):
    f = os.path.join(data_root, 'training.txt')
    lines = open(f).readlines()
    lines = filter(lambda s: selected_dataset in s, lines)
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
        filenames[i] = os.path.join(data_root, ss[0].replace('\\', os.path.sep))
        genders[i], smiles[i], glasses[i], poses[i] = map(int, ss[-4:])
        landmarks[i, np.arange(10)] = tuple(map(float, ss[1:-4]))
    print()
    print(filenames[0])
    print(genders[0], smiles[0], glasses[0], poses[0])
    print(landmarks[0])
    print()

    assert not os.path.exists(out_dir)
    os.makedirs(out_dir)
    f = os.path.join(out_dir, 'filenames.txt')
    with open(f, 'wb') as out:
        for f in filenames:
            line = '%s\n' % f
            out.write(line.encode('utf-8'))
    f = os.path.join(out_dir, 'genders.npy')
    genders.tofile(f)
    f = os.path.join(out_dir, 'smiles.npy')
    smiles.tofile(f)
    f = os.path.join(out_dir, 'glasses.npy')
    glasses.tofile(f)
    f = os.path.join(out_dir, 'poses.npy')
    poses.tofile(f)
    f = os.path.join(out_dir, 'landmarks.npy')
    landmarks.tofile(f)

    return filenames


def process_images(data_root, selected_dataset, listed_filenames, out_dir):
    images_dir = '%s/%s/' % (data_root, selected_dataset)
    ff = os.listdir(images_dir)
    ff = map(lambda f: os.path.join(images_dir, f), ff)
    listed_filenames_set = set(listed_filenames)
    ff = filter(lambda f: f in listed_filenames_set, ff)
    ff = sorted(ff)
    assert ff == listed_filenames

    for i, image_fn in tqdm(enumerate(ff), total=len(ff)):
        im = imageio.imread(image_fn)
        x = np.array(im)
        x = x.transpose([2, 0, 1])
        x = zoom(x, (1, 64 / x.shape[1], 64 / x.shape[2]))


def run(flags):
    filenames = process_labels(flags.data_root, flags.selected_dataset,
                               flags.out_dir)
    process_images(flags.data_root, flags.selected_dataset, filenames,
                   flags.out_dir)


if __name__ == '__main__':
    run(parse_flags())

import imageio
import numpy as np
import os



data_dir = 'data'
dataset_name = 'lfw_5590'

f = os.path.join(data_dir, 'training.txt')
lines = open(f).readlines()
lines = filter(lambda s: dataset_name in s, lines)
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
    filenames[i] = os.path.join(data_dir, ss[0].replace('\\', os.path.sep))
    genders[i], smiles[i], glasses[i], poses[i] = map(int, ss[-4:])
    landmarks[i, np.arange(10)] = tuple(map(float, ss[1:-4]))
print()
print(filenames[0])
print(genders[0], smiles[0], glasses[0], poses[0])
print(landmarks[0])
print()
filenames_set = set(filenames)

images_dir = '%s/%s/' % (data_dir, dataset_name)
ff = os.listdir(images_dir)
ff = map(lambda f: os.path.join(images_dir, f), ff)
ff = filter(lambda f: f in filenames_set, ff)
ff = sorted(ff)

"""
f = ff[0]
print(f)
im = imageio.imread(f)
x = np.array(im)
x = x.transpose([2, 0, 1])
print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())
"""

assert filenames == ff

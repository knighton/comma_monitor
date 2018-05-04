import imageio
import numpy as np
import os


dataset_name = 'lfw_5590'
images_dir = 'data/%s/' % dataset_name
ff = os.listdir(images_dir)
ff = sorted(map(lambda f: os.path.join(images_dir, f), ff))
f = ff[0]
print(f)
im = imageio.imread(f)
x = np.array(im)
x = x.transpose([2, 0, 1])
print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())


f = 'data/training.txt'
lines = open(f).readlines()
lines = filter(lambda s: dataset_name in s, lines)
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
    filenames[i] = ss[0]
    genders[i], smiles[i], glasses[i], poses[i] = map(int, ss[-4:])
    landmarks[i, np.arange(10)] = tuple(map(float, ss[1:-4]))
    break
print()
print(filenames[0])
print(genders[0], smiles[0], glasses[0], poses[0])
print(landmarks[0])
print()

import imageio
import numpy as np
import os


images_dir = 'data/lfw_5590/'
ff = os.listdir(images_dir)
ff = list(map(lambda f: os.path.join(images_dir, f), ff))
print(ff)
f = ff[0]
print(f)
im = imageio.imread(f)
x = np.array(im)
print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())

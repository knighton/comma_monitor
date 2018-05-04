import imageio
import numpy as np
import os
from scipy.ndimage.interpolation import zoom


f = '/home/frak/Documents/ace.jpg'
print(f)
im = imageio.imread(f)
x = np.array(im)
x = x.transpose([2, 0, 1])
print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())
x = zoom(x, (3, 64 / x.shape[1], 50 / x.shape[2]))
print(x.shape, x.dtype, x.min(), x.max(), x.mean(), x.std())

import numpy as np
from scipy.ndimage.filters import uniform_filter1d

a = [0.3, 1.1, 1.4, 3, 2.8, 2.8, 2, 3]
b = [[2, 1.5, 3.5], [2.5, 1, 3.5], [3, 0.5, 3.5], [4, 0.5, 3.5]]
N = 3

y = uniform_filter1d(b, axis=0, size=N, mode='nearest')
print(y)
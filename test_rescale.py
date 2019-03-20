from dface.core.detect import resize_image
import torch
import time
import cv2
import numpy as np
a = torch.ones(10, 3, 100, 100)
times = 1000
start = time.time()
end = time.time()
print(f'Duration : {end-start}')


start = time.time()
[resize_image(a, 0.25) for i in range(times)]
end = time.time()
print(f'Duration : {end-start}')

a = a.cuda()
start = time.time()
[resize_image(a, 0.25) for i in range(times)]
end = time.time()
print(f'Duration : {end-start}')

a = np.ones([100, 100, 3])
start = time.time()
[cv2.resize(a, (25, 25)) for i in range(times*10)]
end = time.time()
print(f'Duration : {end-start}')

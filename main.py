import numpy as np
import matplotlib.pyplot as plt
import sys

brown = np.load('./Arshia7_Brown_f819.npz')
beer = np.load('./Arshia8_Beer_17b5.npz')

brown_x = brown['x']
brown_y = brown['y']
beer_x = beer['x']
beer_y = beer['y']

print(brown_x.shape, brown_y.shape)
print(beer_x.shape, beer_y.shape)

zeros = np.zeros(shape=(1, 480, 640, 90))

brown_x = np.concat([brown_x, zeros], axis=0)
beer_x = np.concat([beer_x, zeros], axis=0)

brown_summed = np.sum(brown_x, axis=-1)
beer_summed = np.sum(beer_x, axis=-1)

brown_summed[brown_summed > 0] = 1.0
beer_summed[beer_summed > 0] = 1.0

brown_summed = np.moveaxis(brown_summed, 0, -1)
beer_summed = np.moveaxis(beer_summed, 0, -1)

plt.imsave('brown_summed.jpg', brown_summed)
plt.imsave('beer_summed.jpg', beer_summed)

brown_x = np.moveaxis(brown_x, -1, 0)
beer_x = np.moveaxis(beer_x, -1, 0)

print(brown_x.shape, beer_x.shape)

for i, frame in enumerate(brown_x):
    new_frame = np.moveaxis(frame, 0, -1)
    new_frame[new_frame > 0] = 1.0
    plt.imsave(f'brown/frame_{i}.jpg', new_frame)

for i, frame in enumerate(beer_x):
    new_frame = np.moveaxis(frame, 0, -1)
    new_frame[new_frame > 0] = 1.0
    plt.imsave(f'beer/frame_{i}.jpg', new_frame)

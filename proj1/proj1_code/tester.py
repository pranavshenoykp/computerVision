import numpy as np
import part1 as p

filter = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
)

channel_img = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
)

img = np.zeros((4, 4, 3), dtype=np.uint8)
img[:, :, 0] = channel_img
img[:, :, 1] = channel_img
img[:, :, 2] = channel_img

filtered_img = p.my_conv2d_numpy(img, filter)

print(img)

print(filtered_img)
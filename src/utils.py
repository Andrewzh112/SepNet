from glob import glob
import os
from PIL import Image
from random import shuffle
import numpy as np


def filter_images(min_dim=224, image_path='images'):
    for image in glob(os.path.join(image_path, '*')):
        width, height = Image.open(image).size
        if height < min_dim or width < min_dim:
            os.remove(image)


def pair_images(batch_size):
    assert (batch_size % 2) == 0
    indices = list(range(batch_size))
    shuffle(indices)
    for i in range(batch_size // 2):
        yield indices[i*2], indices[i*2 + 1]


def ranom_uniform_pair(size, low=0.3, high=0.7):
    portions = np.random.uniform(low=low, high=high, size=size)
    return [(portion, 1-portion) for portion in portions]

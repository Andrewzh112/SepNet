from glob import glob
import os
from PIL import Image
import random
import numpy as np
import torch
import logging
from torchvision import transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def filter_images(min_dim=224, image_path='images'):
    for image in glob(os.path.join(image_path, '*')):
        width, height = Image.open(image).size
        if height < min_dim or width < min_dim:
            os.remove(image)


def pair_indices(batch_size):
    assert (batch_size % 2) == 0
    indices = list(range(batch_size))
    random.shuffle(indices)
    for i in range(batch_size // 2):
        yield indices[i*2], indices[i*2 + 1]


def ranom_uniform_pair(batch_size, low=0.3, high=0.7):
    for _ in range(batch_size):
        portion = np.random.uniform(low=low, high=high, size=1)[0]
        if portion == 0.5:
            adj = np.random.uniform(low=low, high=high, size=1)[0] * 0.1
            portion += adj
        yield (portion, 1-portion)


def mix_images(images):
    batch_size = images.shape[0]
    pairs = pair_indices(batch_size)
    ratios = list(ranom_uniform_pair(batch_size))
    image_pairs, mixed_images = [], []
    for (i, j), ratio in zip(pairs, ratios):
        image0, image1 = images[i], images[j]
        pair = torch.stack([image0, image1], dim=0)
        image_pairs.append(pair)
        r0, r1 = ratio
        mix_image = r0 * image0 + r1 * image1
        mixed_images.append(mix_image)
    image_pairs = torch.stack(image_pairs, dim=0)
    mixed_images = torch.stack(mixed_images, dim=0)
    return image_pairs, mixed_images, ratios


def unstack_images(constructed_images, statistics):
    M, S = statistics
    unnormalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0., 0., 0.], std=[1/s for s in S]),
            transforms.Normalize(mean=[-m for m in M], std=[1., 1., 1.])
        ]
    )
    images = []
    for ci in constructed_images:
        image0 = unnormalize(ci[:3, :, :])
        image1 = unnormalize(ci[3:, :, :])
        images.append(image0)
        images.append(image1)
    images = torch.stack(images, dim=0)
    return images

from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from torchvision import transforms
from random import shuffle
from sepnet.utils import filter_images
import numpy as np
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


class SepDataset(Dataset):
    def __init__(self,
                 statistics=None,
                 transform=None,
                 img_path='images',
                 filter_imgs=False,
                 image_size=224):
        super().__init__()
        if filter_imgs:
            filter_images(min_dim=image_size, image_path=img_path)
        self.images = glob(os.path.join(img_path, '*'))
        shuffle(self.images)
        # M = [0.48572235511288586, 0.4533838840736244, 0.41519041094505277]
        # S = [0.2619148023576319, 0.2551473237582999, 0.2580172359016024]
        if statistics is None:
            M, S = self.dataset_stats()
            print('New dataset means M =', M)
            print('New dataset stds S =', S)
        else:
            assert isinstance(statistics, (tuple, list)
                              ) and len(statistics) == 2
            M, S = statistics
        self.statistics = (M, S)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(M, S)
                ]
            )

    def dataset_stats(self):
        """[Summary]

        Returns:
            list: list of means for R, G, B channels for whole image dataset (normalized by 255)
            list: list of standard deviations for R, G, B channels for whole image dataset (normalized by 255)
        """
        M = self._calculate_pixel_mean()
        S = self._calculate_pixel_std(M)
        return list(map(lambda x: x/255, M)), list(map(lambda x: x/255, S))

    def _calculate_pixel_mean(self):
        R, G, B = 0, 0, 0
        pixels = 0
        for img in tqdm(self.images):
            image = np.asarray(Image.open(img))
            pixels += image.shape[0] * image.shape[1]

            R += np.sum(image[:, :, 0])
            G += np.sum(image[:, :, 1])
            B += np.sum(image[:, :, 2])

        R_mean, G_mean, B_mean = R / pixels, G / pixels, B / pixels
        return [R_mean, G_mean, B_mean]

    def _calculate_pixel_std(self, M):
        R, G, B = 0, 0, 0
        pixels = 0
        R_mean, G_mean, B_mean = M
        for img in tqdm(self.images):
            image = np.asarray(Image.open(img))
            pixels += image.shape[0] * image.shape[1]

            R += np.sum((image[:, :, 0] - R_mean) ** 2)
            G += np.sum((image[:, :, 1] - G_mean) ** 2)
            B += np.sum((image[:, :, 2] - B_mean) ** 2)

        R_std, G_std, B_std = np.sqrt(
            R / pixels), np.sqrt(G / pixels), np.sqrt(B / pixels)
        return [R_std, G_std, B_std]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        return self.transform(img)


def get_loader(batch_size=16, shuffle=True, num_workers=8):
    return DataLoader(
        SepDataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

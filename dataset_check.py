import glob

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from utils import RandomTransWrapper

class CarlaH5Dataset(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", datapoints_per_file=200):
        self.data_dir = data_dir
        self.files = glob.glob(data_dir+'*.h5')
        self.datapoints_per_file = datapoints_per_file
        self.train_eval_flag = train_eval_flag

        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.LinearContrast(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.datapoints_per_file * len(self.files)

    def __getitem__(self, idx):
        data_idx = idx // self.datapoints_per_file
        file_idx = idx % self.datapoints_per_file
        file_name = self.files[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'])[file_idx]
            img = self.transform(img)
            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)
            # 2 Follow lane, 3 Left, 4 Right, 5 Straight
            # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
            cmd = int(target[24])-2
            # Steer, Gas, Brake (0,1, focus on steer loss)
            target_v = np.zeros((4, 3), dtype=np.float32)
            target_v[cmd, :] = target[:3]
            # in km/h, <90
            speed = np.array([target[10]/ 90.0, ]).astype(np.float32)
            mask = np.zeros((4, 3), dtype=np.float32)
            mask[cmd, :] = 1

        return img, speed, target_v.flatten(), mask.flatten(),
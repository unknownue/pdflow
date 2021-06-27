
import h5py as h5
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.transform import AddRandomNoise, AddNoise, AddNoiseForEval
from dataset.transform import RandomScale, IdentityTransform
from dataset.transform import RandomRotate


class DMRDenoiseDataset(Dataset):

    def __init__(self, h5paths, dataset_name, normal_name='normal', batch_size=1, transform=None, random_get=False, subset_size=1):
        super(DMRDenoiseDataset, self).__init__()

        pointclouds = []
        normals = []

        for path in h5paths:
            h5file = h5.File(path, mode='r')
            pointclouds.append(h5file[dataset_name])
            if normal_name is not None:
                normals.append(h5file[normal_name])
        
        self.pointclouds = np.concatenate(pointclouds, axis=0)
        self.normals = np.concatenate(normals, axis=0) if normal_name is not None else None

        self.transform = transform
        self.batch_size = batch_size
        self.random_get = random_get
        self.subset_size = subset_size
 
    def __len__(self):
        if self.subset_size is not None:
            return self.subset_size
        else:
            return self.pointclouds.shape[0]
    
    def __getitem__(self, index):
        if self.random_get:
            index = np.random.randint(0, self.pointclouds.shape[0] - 1)

        item = {
            'pos': torch.FloatTensor(self.pointclouds[index]),
        }
        if self.normals is not None:
            item['normal'] = torch.FloatTensor(self.normals[index])

        if self.transform is not None:
            item = self.transform(item)

        return item



class DMRDenoiseDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(DMRDenoiseDataModule, self).__init__()
        self.cfg = cfg

    def train_dataloader(self):
        # noisifier
        noise_l = self.cfg.noise_low
        noise_h = self.cfg.noise_high
        if noise_h > noise_l:
            noisifier = AddRandomNoise(std_range=[noise_l, noise_h])
            print(f'[INFO] Using random noise level [{noise_l}, {noise_h}]')
        else:
            noisifier = AddNoise(std=self.cfg.noise_low)

        # Scaling augmentation
        if self.cfg.aug_scale:
            print('[INFO] Scaling augmentation Enable')
            # anisotropic scaling doesn't change the direction of normal vectors
            scaler = RandomScale([0.8, 1.2], attr=['pos', 'clean'])
        else:
            print('[INFO] Scaling augmentation Disable')
            scaler = IdentityTransform()

        t = transforms.Compose([
            noisifier,
            # rotate normal vectors as well
            RandomRotate(degrees=30, attr=['pos', 'clean', 'normal']),
            scaler,
        ])
        
        if isinstance(self.cfg.datasets, list) and len(self.cfg.datasets) > 1:
            print('[INFO] Using multiple datasets for training.')
            dataset = DMRDenoiseDataset(self.cfg.datasets, 'train', normal_name='train_normal', batch_size=self.cfg.batch_size, transform=t, random_get=True, subset_size=self.cfg.subset_size)
        else:
            dataset = DMRDenoiseDataset([self.cfg.datasets], 'train', normal_name='train_normal', batch_size=self.cfg.batch_size, transform=t, random_get=None, subset_size=None)
        
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        noisifier = AddNoiseForEval(stds=[0.01, 0.03, 0.08])
        t = transforms.Compose([noisifier])

        self.val_noisy_item_keys = noisifier.keys
        
        if isinstance(self.cfg.datasets, list) and len(self.cfg.datasets) > 1:
            dataset_path = [self.cfg.datasets[0]]
            print('[INFO] Validation dataset %s' % dataset_path)
        else:
            dataset_path = [self.cfg.datasets]
        
        dataset = DMRDenoiseDataset(dataset_path, 'val', normal_name='val_normal', batch_size=self.cfg.batch_size, transform=t, random_get=None, subset_size=None)

        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=False, drop_last=True, num_workers=self.cfg.num_workers)

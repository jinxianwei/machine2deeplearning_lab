import os
import random
from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

def read_csv_data(data_path: str):
    pandas_data = pd.read_csv(data_path)
    data = np.array(pandas_data.values)
    name = list(pandas_data.columns)
    name_dict = {}
    for i in range(len(name)):
        name_dict[i] = name[i]
    
    return name_dict, data

class Npv_Dataset():
    def __init__(self, data_path, transform) -> None:
        self.data_path = data_path
        self.transform = transform
        self.name_dict, self.data = read_csv_data(self.data_path)
        self.feature = self.data[:, 0:-1]
        self.target = self.data[:, -1]
        
    def __getitem__(self, idx):
        data = self.feature[idx]
        target = self.target[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, target
    
    def __len__(self):
        return len(self.target)
    
class Npv_DataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path: str = None,
                 split_test: float = 0.2,
                 batch_size: int = 8,
                 num_workers: int = 0,
                 transform: Optional['transforms.Compose'] = None,
                 random_seed: int = 2023) -> None:
        super().__init__()
        self.data_path = data_path
        self.split_test = split_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.random_seed = random_seed
        
    def setup(self, stage: str) -> None:
        MyDataset = Npv_Dataset(self.data_path, self.transform)
        if stage == 'fit':
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            test_size = len(MyDataset) * self.split_test
            train_size = len(MyDataset) - test_size
            # TODO 随机划分训练和测试集，对于分类任务可能出现类别不均衡划分问题
            self.train_dataset, self.test_dataset = random_split(MyDataset, [int(train_size), int(test_size)])
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, 
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader
    
    def val_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers)
        return test_loader
        
        
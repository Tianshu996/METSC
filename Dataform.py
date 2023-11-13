#!/usr/bin/python3.8
# @Author  : Tianshu Zheng
# @Email   : zhengtianshu996@gmail.com
# @File    : test.py
# @Software: PyCharm

'''
This is the dataform code of our publication in Medical Image Analysis (T. Zheng et al. “A microstructure estimation Transformer inspired by sparse representation for diffusion MRI,” Med. Image Anal., vol. 86, p. 102788, 2023)
'''

import torch
import torch.utils.data as Data
import numpy as np
import h5py

class Mydataset(Data.Dataset):

    def __init__(self, datafile, normalize=False):

        self.data = h5py.File(datafile,'r')

        self.data = self.data['data'][:]

        self.data = torch.from_numpy(self.data)

        self.data = self.data.permute(1, 0, 3, 2)

        if normalize:

            self.data = self.norm()

    def norm(self):
        self.mu = np.mean(self.data, axis=0)

        self.std = np.std(self.data, axis=0)

        return (self.data - self.mu)/self.std

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        item_A = self.data[index,0:60,:,:]

        item_B = self.data[index,-3:,1,1].unsqueeze(1)
        
        return {'A': item_A, 'B': item_B}
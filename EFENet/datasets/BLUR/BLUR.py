import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from scipy import io as sio
from torch.utils import data
from PIL import Image, ImageOps

class DATA_api(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None, if_dedata=False):
        self.img_path = data_path + '/img'
        self.if_dedata = if_dedata
        if self.if_dedata:
            self.gt_path = data_path + '/deden'
        else:
            self.gt_path = data_path + '/label'
            self.confuse_gt_path = data_path + '/label_adaptive'
        self.data_files = [filename for filename in os.listdir(self.img_path)
                           if os.path.isfile(os.path.join(self.img_path, filename))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den, den_adaptive = self.read_image_and_gt(fname)
        img=img.resize((320,320),Image.ANTIALIAS)
        den = den.resize((320, 320), Image.ANTIALIAS)
        den_adaptive = den_adaptive.resize((320, 320), Image.ANTIALIAS)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
            den_adaptive = self.gt_transform(den_adaptive)
        return img, den, den_adaptive

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        den = Image.open(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.bmp'))
        den_adaptive = Image.open(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.bmp'))
        return img, den, den_adaptive

    def get_num_samples(self):
        return self.num_samples

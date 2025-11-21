#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR = "/root/pytorch_project/metal_am_strain_prediction/3D_U_net/dataset"
NORM_IN  = os.path.join(BASE_DIR, "norm_input")
NORM_OUT = os.path.join(BASE_DIR, "norm_output")

TRAIN_LEN = 144   # 前n个用于训练,修改后记得保存

class NPYDataset(Dataset):
    def __init__(self, base_dir=BASE_DIR, train=True):   
        self.input_dir  = NORM_IN
        self.output_dir = NORM_OUT
        all_files = sorted(glob.glob(os.path.join(self.input_dir, "*.npy")))
        if not all_files:
            raise RuntimeError("❌ 未找到任何归一化数据，请先运行 preprocess_data.py")

        # 切片划分
        if train:
            self.files = all_files[:TRAIN_LEN]
        else:
            self.files = all_files[TRAIN_LEN:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.load(self.files[idx])).float()   # [1,C,64,64,64]
        y_name = os.path.join(self.output_dir, os.path.basename(self.files[idx]))
        y = torch.from_numpy(np.load(y_name)).float()            # [1,*,64,64,64]
        return x.squeeze(0), y.squeeze(0)      # 返回 (C,64,64,64) 和 (*,64,64,64)
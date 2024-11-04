import sys
import cv2
from google.colab.patches import cv2_imshow
import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle
import random
import copy

import os
import shutil
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

from utils import *
from naturaldesign.naturaldesign import NaturalDesign
sys.path.append("..")

class SCEGRAM(Dataset): 
    def __init__(self, info_dir, context_dir, target_dir, context_size, target_size, is_transform=True): 
        df_info = pd.read_excel(info_dir)
        self.df_info = df_info[df_info.obj_name != 'XXX']
        self.image_width = self.df_info.iloc[0, :]['sce_width']
        self.image_height = self.df_info.iloc[0, :]['sce_height']
        self.target_dic = {filename[3:-4]: filename for filename in os.listdir(target_dir)}
        self.context_dir = context_dir
        self.target_dir = target_dir
        self.is_transform = is_transform

        self.context_transform = transforms.Compose([
            transforms.Resize(context_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

        self.target_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) 

    def __len__(self):
        return len(self.df_info)
    
    def __getitem__(self, idx):
        idx_info = self.df_info.iloc[idx, :]
        context_file = os.path.join(self.context_dir, idx_info['sce_file_name'])
        target_file = os.path.join(self.target_dir, self.target_dic[idx_info['obj_name']])
        x_center, y_center, w, h = idx_info['obj_x_center'], idx_info['obj_y_center'], idx_info['obj_width'], idx_info['obj_height']
        bbox_relative = [(x_center-0.5*w)/self.image_width, (y_center-0.5*h)/self.image_height, w/self.image_width, h/self.image_height]
        img_category = idx_info['sce_file_name'][:-4].split('_')[-1]
        
        # process context img, target img
        img = cv2.imread(context_file)
        target = cv2.imread(target_file)
        
        # calculate the bounding box tensor
        bbox_relative = torch.tensor(bbox_relative)

        if self.is_transform:
            img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            target_PIL = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
            img = self.context_transform(img_PIL)
            target = self.target_transform(target_PIL)

        return img, target, bbox_relative, img_category
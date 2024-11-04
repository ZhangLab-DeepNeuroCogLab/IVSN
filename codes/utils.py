import sys
import json
import yaml
import pathlib
import pickle
import random
import copy

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import os
import os.path as osp

import PIL
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
import cv2 as cv
from google.colab.patches import cv2_imshow
from ml_collections import ConfigDict

# extract info of pic for further analysis
def pic_analysis(dataset, index):
    image, target_image, bbox_relative, label, annotation_id = dataset[index]
    width, height = dataset.image_size
    x_min, y_min, w, h = int(bbox_relative[0]*width), int(bbox_relative[1]*height), int(bbox_relative[2]*width), int(bbox_relative[3]*height)
    label_name = dataset.idx2label[label]

    return image, target_image, (x_min, y_min, w, h), label_name
    
def pic_analysis_scegram(dataset, index, image_size):
    image, target_image, bbox_relative, category = dataset[index]
    width, height = image_size
    x_min, y_min, w, h = int(bbox_relative[0]*width), int(bbox_relative[1]*height), int(bbox_relative[2]*width), int(bbox_relative[3]*height)

    return image, target_image, (x_min, y_min, w, h)

def target_rec(image_tensor, cordinates):
    image = transforms.ToPILImage()(image_tensor)
    x_min, y_min, w, h = cordinates
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min,x_min+w, y_min+h], outline=(255,0,0)) 
    return image

def bbox_cordinates(bbox_relative, width, height):
    x_min, y_min, w, h = int(bbox_relative[0]*width), int(bbox_relative[1]*height), int(bbox_relative[2]*width), int(bbox_relative[3]*height)

    return (x_min, y_min, w, h)

def searchProcess(tg_xy, attentionMap, image_size, size):
    mask_size = size
    tg_x, tg_y, w, h = tg_xy
    tg_xmax, tg_ymax = tg_x + w, tg_y + h 
    attentionMap[0, int(image_size[0]//2), int(image_size[1]//2)] = 1000
    maxSearch = 999

    count = 0 # num of search times of model following human's cordinates
    while True:
        attenNP = attentionMap[0,:,:].detach().numpy()
        y_fix, x_fix = np.unravel_index(attenNP.argmax(), attenNP.shape)
        count += 1

        # max search times, if exceeded, break
        if count >= maxSearch:
            break

        x_max_s, x_min_s, y_max_s, y_min_s = min(x_fix+mask_size//2, image_size[1]-1), max(x_fix-mask_size//2, 0), min(y_fix+mask_size//2, image_size[0]-1), max(y_fix-mask_size//2, 0)

        if x_max_s < tg_x or x_min_s > tg_xmax or y_max_s < tg_y or y_min_s > tg_ymax:
            attentionMap[0, y_min_s:y_max_s+1, x_min_s:x_max_s+1] = 0
        else:
            break
    return count

def searchProcesswithPath(tg_xy, attentionMap, image_size, size):
    searchPath = []
    mask_size = size
    tg_x, tg_y, w, h = tg_xy
    tg_xmax, tg_ymax = tg_x + w, tg_y + h 
    attentionMap[0, int(image_size[0]//2), int(image_size[1]//2)] = 1000
    maxSearch = 999

    count = 0 # num of search times of model following human's cordinates
    while True:
        attenNP = attentionMap[0,:,:].detach().numpy()
        y_fix, x_fix = np.unravel_index(attenNP.argmax(), attenNP.shape)
        searchPath.append([x_fix, y_fix])
        count += 1

        # max search times, if exceeded, break
        if count >= maxSearch:
            break

        x_max_s, x_min_s, y_max_s, y_min_s = min(x_fix+mask_size//2, image_size[1]-1), max(x_fix-mask_size//2, 0), min(y_fix+mask_size//2, image_size[0]-1), max(y_fix-mask_size//2, 0)

        if x_max_s < tg_x or x_min_s > tg_xmax or y_max_s < tg_y or y_min_s > tg_ymax:
            attentionMap[0, y_min_s:y_max_s+1, x_min_s:x_max_s+1] = 0
        else:
            break
    return count, searchPath

def model_performance(search_list, image_num):
    search_counter = pd.value_counts(search_list)
    # max_search_times = max(search_list)
    max_search_times = 99
    sum_found_model = 0
    accu_model_performance = [0 for _ in range(max_search_times + 1)]
    for i in range(1, max_search_times + 1):
        try:
            sum_found_model += search_counter[i]
            accu_model_performance[i] = sum_found_model / image_num
        except KeyError:
            accu_model_performance[i] = sum_found_model / image_num
    
    return accu_model_performance
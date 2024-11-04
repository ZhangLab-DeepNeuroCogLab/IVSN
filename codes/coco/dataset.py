import os
import json
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

from PIL import Image
import pickle
import random
from collections import OrderedDict, Counter

class COCODataset(Dataset):
    """
    Dataset to load data in COCO-style format and provide samples corresponding to individual objects.
    A sample consists of a target image (cropped to the objects bounding box), a context image (entire image),
    the bounding box coordinates of the target object ([xmin, ymin, w, h]) relative to the image size (e.g., (0.5,0.5)
    are the coords of the point in the middle of the image) and a label in [0,num_classes].
    """

    def __init__(self, annotations_file, image_dir, image_size, category_dic_dir=None, idx2label=None, normalize_means=None, normalize_stds=None):
        """
        Args:
            annotations_file: path to COCO-style annotation file (.json)
            image_dir: path to the image folder
            image_size: desired size of the sample images, either a tuple (w,h) or an int if w=h
            idx2label: If a particular mapping between index and label is desired. Format: {idx: "labelname"}.
            normalize_means: List of means for each channel. Set None to disable normalization.
            normalize_stds: List of standard deviations for each channel. Set None to disable normalization.
        """
        
        self.image_dir = image_dir
        if category_dic_dir is not None: 
            with open(category_dic_dir, "rb") as tf:
                self.category_idx_dict = pickle.load(tf)
        else:
            self.category_idx_dict = {} 
            
        self.image_size = image_size if type(image_size) == tuple else (image_size, image_size)

        with open(annotations_file) as f:
            self.coco_dict = json.load(f, object_pairs_hook=OrderedDict)

        self.annotations = self.coco_dict["annotations"]
        
        self.id2file = {}
        for i in self.coco_dict["images"]:
            self.id2file[i["id"]] = os.path.join(image_dir, i["file_name"])

        self.id2label = {} # maps label id to label name
        self.id2idx = {} # maps label id to index in 1-hot encoding
        if idx2label is None:
            self.idx2label = {} # maps index in 1-hot encoding to label name
            for idx, i in enumerate(self.coco_dict["categories"]):
                self.id2label[i["id"]] = i["name"]    
                self.idx2label[idx] = i["name"]
                self.id2idx[i["id"]] = idx
        else:
            assert(len(self.coco_dict["categories"]) == len(idx2label)), "Number of categorires in the annotation file does not agree with the number of categories in the custom idx2label mapping."
            
            self.idx2label = idx2label # maps index in 1-hot encoding to label name
            label2idx = {label: idx for idx, label in self.idx2label.items()}
            for i in self.coco_dict["categories"]:
                self.id2label[i["id"]] = i["name"]
                self.id2idx[i["id"]] = label2idx[i["name"]]
        
        self.NUM_CLASSES = len(self.id2label)

        # count annotations per class
        self.annotation_counts = Counter([a["category_id"] for a in self.annotations])
        self.annotation_counts = {self.id2idx[k]: v for k, v in self.annotation_counts.items()}
        self.named_annotation_counts = {self.idx2label[k]: v for k, v in self.annotation_counts.items()}
        self.relative_annotation_counts = np.array([self.annotation_counts[k] for k in sorted(self.annotation_counts.keys())])
        self.relative_annotation_counts = self.relative_annotation_counts / np.sum(self.relative_annotation_counts)
        self.relative_annotation_counts = torch.tensor(self.relative_annotation_counts, dtype=torch.float) # convert to tensor to simplify usage for reweighting
        
        print("-------------------------------\nAnnotation Counts\n-------------------------------")
        for k, v in self.named_annotation_counts.items():
            print("{0:20} {1:10}".format(k, v))
        print("{0:20} {1:10}".format("Total", len(self.annotations)))
        print("-------------------------------\n")

        if normalize_means is not None and normalize_stds is not None:
            self.normalize_means = normalize_means
            self.normalize_stds = normalize_stds
            self.normalize = True
        else:
            self.normalize = False

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]

        # load image
        image = Image.open(self.id2file[annotation["image_id"]])
        image = image.convert("RGB")
        
        # compute bounding box coordinates relative to the image size
        xmin, ymin, w, h = annotation["bbox"]
        bbox_relative = torch.tensor([xmin / image.width, ymin / image.height, w / image.width, h / image.height])

        # crop to bounding box for target image
        target_image = image.crop((int(xmin), int(ymin), int(xmin + w), int(ymin + h)))

        # resize
        image = image.resize(self.image_size)
        target_image = target_image.resize(self.image_size)

        # convert to torch tensor
        image = to_tensor(image)
        target_image = to_tensor(target_image)

        # normalize
        if self.normalize:
            image = normalize(image, self.normalize_means, self.normalize_stds)
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)

        label = self.id2idx[annotation["category_id"]]

        return image, target_image, bbox_relative, label


class COCODatasetWithID(COCODataset):
    """
    Provides the same functionality as COCODataset but in addition the id of the annotation is also returned.
    This can be useful for in-depth analysis of the results.
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        annotation_id = self.annotations[idx]["image_id"]
        return (*sample, annotation_id)
    
    
class COCODatasetZeroTarget(COCODataset):
    """
    Provides the same functionality as COCODataset but empty target is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        image, _, bbox_relative, label = super().__getitem__(idx)

        target = torch.zeros(3, self.image_size[0], self.image_size[1])  
        
        return image, target, bbox_relative, label
    
    
class COCODatasetRandom(COCODataset):
    """
    Provides the same functionality as COCODataset but random target crop within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        image, _, bbox_relative, label = super().__getitem__(idx)
        label_name = self.idx2label[label]
        # get the label name
        category_size = len(self.category_idx_dict[label_name]) 
        
        # get the random index within the category
        random_idx = random.randint(0, category_size-1)
        
        # get the annotation according to the random index
        target_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx]]
        image_random = Image.open(self.id2file[target_random_annotation["image_id"]])
        image_random = image_random.convert("RGB") 
        xmin_random, ymin_random, w_random, h_random = target_random_annotation["bbox"]
        target_image = image_random.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random))) 

        # resize the target image
        target_image = target_image.resize(self.image_size)

        # convert to torch tensor
        target_image = to_tensor(target_image)

        # normalize
        if self.normalize:
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)        
        
        return image, target_image, bbox_relative, label
    
    
class COCODatasetMix(COCODataset):
    """
    Provides the same functionality as COCODataset but mix target crop(random or empty) within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        # choice for context and target target
        # context choice: 0 -> random, 1 -> empty, 2 -> original 
        # target choice: 0 -> random, 1 -> empty
        choice_context, choice_target, target_image = random.choice([0, 1, 2]), random.choice([0, 1]), None
        
        # info inherited from the COCODataset
        image, _, bbox_relative, label = super().__getitem__(idx)
        # get the label name
        label_name = self.idx2label[label]
        
        # for test purposes
        # print(label_name, choice_context, choice_target)
        
        # get the random index within the category
        category_size = len(self.category_idx_dict[label_name])       
        random_idx = random.randint(0, category_size-1)
        # get the original position of target in context image
        bbox_og = bbox_relative.tolist()
        xmin_og, ymin_og, w_og, h_og = int(self.image_size[1]*bbox_og[0]), int(self.image_size[0]*bbox_og[1]), int(self.image_size[1]*bbox_og[2]+1), int(self.image_size[0]*bbox_og[3]+1) 
        
        if choice_target == 0 or choice_context == 0:
            # get the annotation according to the random index
            target_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx]]
            image_random = Image.open(self.id2file[target_random_annotation["image_id"]])
            image_random = image_random.convert("RGB") 
            xmin_random, ymin_random, w_random, h_random = target_random_annotation["bbox"]
            target_image = image_random.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random)))
            # resize the target image
            target_image = target_image.resize(self.image_size)
            # convert to torch tensor
            target_image = to_tensor(target_image)
            # normalize
            if self.normalize:
                target_image = normalize(target_image, self.normalize_means, self.normalize_stds)        
        
        if choice_target == 0 and choice_context == 0:
            w, h = min(w_og, min(xmin_og+w_og, self.image_size[1])-xmin_og), min(h_og, min(ymin_og+h_og, self.image_size[0])-ymin_og)
            target_image = transforms.Resize((h, w))(target_image)
            image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = target_image
            target_image = transforms.Resize(self.image_size)(target_image)
            
        elif choice_target == 0 and choice_context == 1: 
            image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = 0
            
        elif choice_target == 0 and choice_context == 2: 
            target_image = transforms.Resize(self.image_size)(target_image)
        
        elif choice_target == 1 and choice_context == 0:
            w, h = min(w_og, min(xmin_og+w_og, self.image_size[1])-xmin_og), min(h_og, min(ymin_og+h_og, self.image_size[0])-ymin_og)
            target_image = transforms.Resize((h, w))(target_image)
            image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = target_image
            target_image = torch.zeros(3, self.image_size[0], self.image_size[1]) 
        
        elif choice_target == 1 and choice_context == 1:
            image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = 0 
            target_image = torch.zeros(3, self.image_size[0], self.image_size[1])
            
        elif choice_target == 1 and choice_context == 2:
            target_image = torch.zeros(3, self.image_size[0], self.image_size[1])     
            
        return image, target_image, bbox_relative, label
    
    
class COCODatasetZeroContextZeroTarget(COCODataset):
    """
    Provides the same functionality as COCODataset but zero target crop and context with zero in target area within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):    
        # info inherited from the COCODataset
        image, _, bbox_relative, label = super().__getitem__(idx)
        label_name = self.idx2label[label]
     
        # get the original position of target in context image
        bbox_og = bbox_relative.tolist()
        xmin_og, ymin_og, w_og, h_og = int(self.image_size[1]*bbox_og[0]), int(self.image_size[0]*bbox_og[1]), int(self.image_size[1]*bbox_og[2]+1), int(self.image_size[0]*bbox_og[3]+1) 
        
        # make target area in context image 0
        image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = 0 
        target_image = torch.zeros(3, self.image_size[0], self.image_size[1])    
            
        return image, target_image, bbox_relative, label
    
class COCODatasetZeroContextRandomTarget(COCODataset):
    """
    Provides the same functionality as COCODataset but random target crop within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        # info inherited from the COCODataset
        image, _, bbox_relative, label = super().__getitem__(idx)
        label_name = self.idx2label[label]
        
        # get the label name
        category_size = len(self.category_idx_dict[label_name])       
        # get the random index within the category
        random_idx = random.randint(0, category_size-1)
        # get the original position of target in context image
        bbox_og = bbox_relative.tolist()
        xmin_og, ymin_og, w_og, h_og = int(self.image_size[1]*bbox_og[0]), int(self.image_size[0]*bbox_og[1]), int(self.image_size[1]*bbox_og[2]+1), int(self.image_size[0]*bbox_og[3]+1) 

        # get the annotation according to the random index
        target_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx]]
        image_random = Image.open(self.id2file[target_random_annotation["image_id"]])
        image_random = image_random.convert("RGB") 
        xmin_random, ymin_random, w_random, h_random = target_random_annotation["bbox"]
        target_image = image_random.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random)))
        # resize the target image
        target_image = target_image.resize(self.image_size)
        # convert to torch tensor
        target_image = to_tensor(target_image)
        # normalize
        if self.normalize:
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)        
        
        # make target area in context image zero    
        image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = 0
            
        return image, target_image, bbox_relative, label
        
class COCODatasetRandomContextRandomTarget(COCODataset):
    """
    Provides the same functionality as COCODataset but random target crop within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        # info inherited from the COCODataset
        image, _, bbox_relative, label = super().__getitem__(idx)
        label_name = self.idx2label[label]
        
        # get the label name
        category_size = len(self.category_idx_dict[label_name])       
        # get the random index within the category
        random_idx_target = random.randint(0, category_size-1)
        random_idx_context = random.randint(0, category_size-1)
        # get the original position of target in context image
        bbox_og = bbox_relative.tolist()
        xmin_og, ymin_og, w_og, h_og = int(self.image_size[1]*bbox_og[0]), int(self.image_size[0]*bbox_og[1]), int(self.image_size[1]*bbox_og[2]+1), int(self.image_size[0]*bbox_og[3]+1) 
        
        # get the annotation according to the random index for target
        target_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx_target]]
        image_random_target = Image.open(self.id2file[target_random_annotation["image_id"]])
        image_random_target = image_random_target.convert("RGB") 
        xmin_random, ymin_random, w_random, h_random = target_random_annotation["bbox"]
        target_image = image_random_target.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random)))
        # resize the target image
        target_image = target_image.resize(self.image_size)
        # convert to torch tensor
        target_image = to_tensor(target_image)
        # normalize
        if self.normalize:
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)  
            
        # get the annotation according to the random index for context
        context_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx_context]]
        image_random_context = Image.open(self.id2file[context_random_annotation["image_id"]])
        image_random_context = image_random_context.convert("RGB") 
        xmin_random, ymin_random, w_random, h_random = context_random_annotation["bbox"]
        context_image = image_random_context.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random)))
        # resize the target image
        context_image = context_image.resize(self.image_size)
        # convert to torch tensor
        context_image = to_tensor(context_image)
        # normalize
        if self.normalize:
            context_image = normalize(context_image, self.normalize_means, self.normalize_stds)       
        
        w, h = min(w_og, min(xmin_og+w_og, self.image_size[1])-xmin_og), min(h_og, min(ymin_og+h_og, self.image_size[0])-ymin_og)
        context_image = transforms.Resize((h, w))(context_image)
        image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = context_image
            
        return image, target_image, bbox_relative, label
    
class COCODatasetRandomContextZeroTarget(COCODataset):
    """
    Provides the same functionality as COCODataset but zero target crop within the same class is returned.
    This can be useful for in-depth attention map generated.
    """

    def __getitem__(self, idx):
        # info inherited from the COCODataset
        image, _, bbox_relative, label = super().__getitem__(idx)
        label_name = self.idx2label[label]
        
        # get the label name
        category_size = len(self.category_idx_dict[label_name])       
        # get the random index within the category
        random_idx = random.randint(0, category_size-1)
        # get the original position of target in context image
        bbox_og = bbox_relative.tolist()
        xmin_og, ymin_og, w_og, h_og = int(self.image_size[1]*bbox_og[0]), int(self.image_size[0]*bbox_og[1]), int(self.image_size[1]*bbox_og[2]+1), int(self.image_size[0]*bbox_og[3]+1) 
        
        # get the annotation according to the random index
        target_random_annotation = self.annotations[self.category_idx_dict[label_name][random_idx]]
        image_random = Image.open(self.id2file[target_random_annotation["image_id"]])
        image_random = image_random.convert("RGB") 
        xmin_random, ymin_random, w_random, h_random = target_random_annotation["bbox"]
        target_image = image_random.crop((int(xmin_random), int(ymin_random), int(xmin_random + w_random), int(ymin_random + h_random)))
        # resize the target image
        target_image = target_image.resize(self.image_size)
        # convert to torch tensor
        target_image = to_tensor(target_image)
        # normalize
        if self.normalize:
            target_image = normalize(target_image, self.normalize_means, self.normalize_stds)        
        
        w, h = min(w_og, min(xmin_og+w_og, self.image_size[1])-xmin_og), min(h_og, min(ymin_og+h_og, self.image_size[0])-ymin_og)
        target_image = transforms.Resize((h, w))(target_image)
        image[:, ymin_og:ymin_og+h_og, xmin_og:xmin_og+w_og] = target_image
        target_image = target_image = torch.zeros(3, self.image_size[0], self.image_size[1])
            
        return image, target_image, bbox_relative, label
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import random
import time 
from PIL import Image, ImageFilter
import PIL.ImageOps
import pandas as pd
import os.path as osp

import json

from .triplet_sampler import RandomIdentitySampler


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform       

    def __len__(self):
        return len(self.dataset)
    
    def _read_image(self, img_path):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.filter(ImageFilter.FIND_EDGES)
                img = 255-np.array(img)
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        return img     

    def __getitem__(self, index):
        img_path, pid, att = self.dataset[index]
        img = self._read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, pid, att
    

class zap50k(Dataset):
    def __init__(self, args):
        super(zap50k, self).__init__()
        with open(args.att_path+'zap50k_shoe_id.json') as json_file:
            self.shoe_id = json.load(json_file)
        with open(args.att_path+'zap50k_shoe_att.json') as json_file:
            self.atts = json.load(json_file)

        self.train, self.train_num = self._process_dir(self.shoe_id, self.atts, relabel=True)
        
    def _process_dir(self, labels, atts, relabel=True):

        pid_container = set()
        for _,v in labels.items():
            pid_container.add(v)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        dataset = []
        for k,v in labels.items():
            img_path = k
            pid = v
            att_all = np.array(atts[k])
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid, att_all))
        
        return dataset, len(pid2label)

  
    
def make_zap_loader(args):
    print('==> Loading data..')
    np.random.seed(0)
    # Data loading code
    transform_gray = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h,args.img_w)),
        transforms.Grayscale(num_output_channels=3),
        
        transforms.RandomAffine(5, shear=5),
        transforms.Pad(10),       
        
        #transforms.RandomRotation(5),
        transforms.RandomCrop((args.img_h,args.img_w)),
        transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),  
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    dataset = zap50k(args)
    
    train_set = ImageDataset(dataset.train, transform_gray)
    
    train_loader = DataLoader(train_set,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            sampler = RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance_per_id),           
            drop_last = True
        )
     
    return train_loader, dataset.train_num
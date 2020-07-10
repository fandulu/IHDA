import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from torchvision import transforms
import random
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import time 

from .market_dataset.market_data_manager import *
from .triplet_sampler import *

def make_market_data_loader(args):
    print('==> Loading data..')
    np.random.seed(1234)
    # Data loading code
    
    transform_gray = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h,args.img_w)),
        transforms.Grayscale(num_output_channels=3),
        transforms.Pad(15),
        transforms.RandomCrop((args.img_h,args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    dataset = Market1501(args.data_dir)
    market_set = ImageDataset(dataset.train, args.att_path, transform_gray)
    
    market_loader = data.DataLoader(market_set, args.batch_size, sampler=RandomIdentitySampler(dataset.train, args.batch_size, args.num_instance), num_workers=args.num_workers, drop_last=True)
    
 
    return market_loader, dataset.num_train_pids


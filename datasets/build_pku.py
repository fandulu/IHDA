import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from torchvision import transforms
import random
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import time 
import random
import pickle


class PKU_train_data(data.Dataset):
    def __init__(self,  transform_gray=None):

        with open('./processed_data/pku/train_data.pkl', 'rb') as f:
            self.train_data = pickle.load(f)   
        self.transform_gray = transform_gray
        self.len = len(self.train_data['color_labels'])
        
    def __getitem__(self, index):
        img_color_1,  l_color_1 = self.train_data['color_imgs'][index][0], self.train_data['color_labels'][index]
        img_color_2,  l_color_2 = self.train_data['color_imgs'][index][1], self.train_data['color_labels'][index]
        img_sketch,  l_sketch = self.train_data['sketch_imgs'][index], self.train_data['sketch_labels'][index]   
        img_color_1 = self.transform_gray(img_color_1)
        img_color_2 = self.transform_gray(img_color_2)
        img_sketch = self.transform_gray(img_sketch)

        return img_color_1, img_color_2, img_sketch, l_color_1, l_color_2, l_sketch

    def __len__(self):
        return self.len
    
class PKU_color_test_data(data.Dataset):
    def __init__(self, transform=None):
    
        self.color_imags = np.load('./processed_data/pku/test_color_imgs.npy')
        self.color_labels = np.load('./processed_data/pku/test_color_labels.npy')
        
        self.transform = transform
        
    def __getitem__(self, index):
        img_color,  l_color = self.color_imags[index], self.color_labels[index]
        img_color = self.transform(img_color)

        return img_color, l_color

    def __len__(self):
        return len(self.color_labels)    

class PKU_sketch_test_data(data.Dataset):
    def __init__(self,  transform=None):
    
        self.sketch_imags = np.load('./processed_data/pku/test_sketch_imgs.npy')
        self.sketch_labels = np.load('./processed_data/pku/test_sketch_labels.npy')
        
        self.transform = transform
        
    def __getitem__(self, index):
        img_sketch,  l_sketch = self.sketch_imags[index], self.sketch_labels[index]
        img_sketch = self.transform(img_sketch)

        return img_sketch, l_sketch

    def __len__(self):
        return len(self.sketch_labels)    
    

def make_data_loader_pku(args):
    print('==> Loading data..')
    np.random.seed(1234)
    # Data loading code
    
    transform_gray_train = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((args.img_h,args.img_w)),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Pad(15),
                        transforms.RandomRotation(15),
                        transforms.RandomCrop((args.img_h,args.img_w)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=.05, contrast=.05, saturation=.05, hue=.05),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ])
   

    transform_gray_test = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Resize((args.img_h,args.img_w)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ])
    
    train_set = PKU_train_data(transform_gray_train)
    test_color_set = PKU_color_test_data(transform_gray_test)
    test_sketch_set = PKU_sketch_test_data(transform_gray_test)
    
    return train_set, test_color_set, test_sketch_set 
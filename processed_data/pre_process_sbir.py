import sys
sys.path.extend(['../'])
from matplotlib import pyplot as plt
import os
import glob
import numpy as np
from PIL import Image, ImageFilter
import PIL.ImageOps
import pickle
from config import Config_sbir


def re_label(lables):
    pid2label = {pid:label for label, pid in enumerate(np.sort(np.unique(lables)))}
    new_labels = [pid2label[label] for label in lables]
    return new_labels

def process_train_data_list(args):
        
    color_path_list = glob.glob(args.data_path + 'train/images/*.jpg')
    sketch_path_list = glob.glob(args.data_path + 'train/sketches/*.png')

    num_imgs = len(color_path_list)
    color_raw_imgs = []
    color_imgs = []
    sketch_imgs = []
    
    labels = []
    for i in range(1,num_imgs+1):
        color_path = args.data_path+'train/images/'+str(i)+'.jpg'
        sketch_path = args.data_path+'train/sketches/'+str(i)+'.png'

        color_img_raw = Image.open(color_path).convert('RGB') 
        color_raw_imgs.append(np.array(color_img_raw))
        
        color_img = color_img_raw.filter(ImageFilter.FIND_EDGES)
        color_imgs.append(255-np.array(color_img))

        sketch_img = Image.open(sketch_path).convert('RGB')  
        sketch_imgs.append(np.array(sketch_img))

        labels.append(i-1)
    
    train_data = {'color_imgs': color_imgs,
                  'raw_color_imgs':color_raw_imgs,
                  'sketch_imgs': sketch_imgs,
                  'labels': labels}
    
    f = open('./sbir/train_data.pkl', 'wb')
    pickle.dump(train_data,f)
    f.close()
    
def process_test_data_list(args):
        
    color_path_list = glob.glob(args.data_path + 'test/images/*.jpg')
    sketch_path_list = glob.glob(args.data_path + 'test/sketches/*.png')

    color_imgs = []
    color_raw_imgs = []
    color_labels = []
    for color_path in color_path_list:
        img_id = color_path.split('/')[-1].split('.')[0]
        img_raw = Image.open(color_path).convert('RGB') 
        color_raw_imgs.append(np.array(img_raw))
        img = img_raw.filter(ImageFilter.FIND_EDGES)
        color_imgs.append(255-np.array(img))
        color_labels.append(int(img_id))
    color_labels = re_label(color_labels)

    sketch_imgs = []
    sketch_labels = []
    for sketch_path in sketch_path_list:
        img_id = sketch_path.split('/')[-1].split('.')[0]
        img = Image.open(sketch_path).convert('RGB')    
        sketch_imgs.append(np.array(img))
        sketch_labels.append(int(img_id))
    sketch_labels = re_label(sketch_labels)
    
    test_data = {'color_imgs':color_imgs,
                  'color_labels':color_labels,
                 'raw_color_imgs':color_raw_imgs,
                  'sketch_imgs':sketch_imgs,
                  'sketch_labels':sketch_labels}
    
    f = open('./sbir/test_data.pkl', 'wb')
    pickle.dump(test_data,f)
    f.close()

def main():
    args = Config_sbir()

    process_train_data_list(args)
    process_test_data_list(args)
    
    print('Train data and test data is generated!')
    
if __name__== "__main__":
    main()
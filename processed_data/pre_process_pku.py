import sys
sys.path.extend(['../'])
from config import Config_pku

import os
import glob
import numpy as np
from PIL import Image, ImageChops
import pickle


def read_txt_file(file_path):
    lines = []
    with open(file_path) as f:
        lines.append(f.read().splitlines() )
    f.close()
    lines = np.hstack(lines)
    return lines

def re_label(lables):
    pid2label = {pid:label for label, pid in enumerate(np.sort(np.unique(lables)))}
    new_labels = [pid2label[label] for label in lables]
    return new_labels

def process_train_data_list(data_dir):
    ind_list = np.load('./pku/train_id.npy')
    ind_list = np.sort(ind_list)
        
    color_path_list = glob.glob(data_dir + 'photo/*.jpg')
    sketch_path_list = glob.glob(data_dir + 'sketch/*.jpg')
      
  
    color_imgs = []
    color_labels = []
    for idx in ind_list:
        imgs = []
        for img_path in color_path_list:
            idx_path = int(img_path.split('/')[-1].split('_')[0])
            if idx_path == idx: 
                img = Image.open(img_path)
                img = img.resize((128, 384), Image.ANTIALIAS)
                imgs.append(np.array(img))          
        color_imgs.append(imgs)
        color_labels.append(idx)
    color_labels = re_label(color_labels)
    
    sketch_imgs = []
    sketch_labels = []
    for idx in ind_list:
        for img_path in sketch_path_list:
            idx_path = int(img_path.split('/')[-1].split('.')[0])
            if idx_path == idx: 
                img = Image.open(img_path)
                img = img.resize((128, 384), Image.ANTIALIAS)
                img = np.array(img)             
                sketch_imgs.append(img)
                sketch_labels.append(idx)  
    sketch_labels = re_label(sketch_labels)
    
    train_data = {'color_imgs':color_imgs,
                  'color_labels':color_labels,
                  'sketch_imgs':sketch_imgs,
                  'sketch_labels':sketch_labels}
    
    #np.save(data_dir+'train_data.npy', train_data)
    f = open('./pku/train_data.pkl', 'wb')
    pickle.dump(train_data,f)
    f.close()
    
def process_test_data_list(data_dir):
    ind_list = np.load('./pku/test_id.npy')     
    ind_list = np.sort(ind_list)
        
    color_path_list = glob.glob(data_dir + 'photo/*.jpg')
    sketch_path_list = glob.glob(data_dir + 'sketch/*.jpg')

    color_imgs = []
    color_labels = []
    for idx in ind_list:
        for img_path in color_path_list:
            idx_path = int(img_path.split('/')[-1].split('_')[0])
            if idx_path == idx: 
                img = Image.open(img_path)
                img = img.resize((128, 384), Image.ANTIALIAS)
                img = np.array(img)        
                color_imgs.append(img)
                color_labels.append(idx)
    color_labels = re_label(color_labels)
    
    np.save('./pku/test_color_imgs.npy', color_imgs)
    np.save('./pku/test_color_labels.npy', color_labels)
    
    sketch_imgs = []
    sketch_labels = []
    for idx in ind_list:
        for img_path in sketch_path_list:
            idx_path = int(img_path.split('/')[-1].split('.')[0])
            if idx_path == idx: 
                img = Image.open(img_path)
                img = img.resize((128, 384), Image.ANTIALIAS)
                img = np.array(img)             
                sketch_imgs.append(img)
                sketch_labels.append(idx)  
    sketch_labels = re_label(sketch_labels)
    
    np.save('./pku/test_sketch_imgs.npy', sketch_imgs)
    np.save('./pku/test_sketch_labels.npy', sketch_labels)



def main():
    args = Config_pku()
    
    style_dir = args.data_dir + 'styleAnnotation/'
    style_list = glob.glob(style_dir + '*.txt')
    sample_ind = {}
    for style_path in style_list:
        style_clc = style_path.split('/')[-1].split('_')[0]
        lines = read_txt_file(style_path)
        index = [int(line) for line in lines]
        sample_ind[style_clc] = index
        
    train_id = []
    test_id = []
    split_pos = [34, 15, 60, 25, 16] # given by 'Cross-Domain Adversarial Feature Learning for Sketch Re-identification'
    for style_clc, split in zip(sample_ind,split_pos):
        all_ind = np.random.permutation(sample_ind[style_clc])
        train_id += list(all_ind[:split])
        test_id += list(all_ind[split:])

    np.save('./pku/train_id.npy', train_id)
    np.save('./pku/test_id.npy', test_id)
    
    print('Train list and test list is generated!')
    
    process_train_data_list(args.data_dir)
    process_test_data_list(args.data_dir)
    
    print('Train data and test data is generated!')
    
if __name__== "__main__":
    main()
from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from config import Config_market, Config_pku
from utils import *
from datasets import *
from net.model import *
from engine.pku_market_trainer import *

import random
random.seed(1234)

###############Init Setting#########################################
args = Config_pku()
checkpoint_path = args.model_path    
    
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

###############Init Setting##########################################



###############Load Data##############################################
market_args = Config_market()
market_loader, n_market_clc = make_market_data_loader(market_args)

train_set, test_color_set, test_sketch_set = make_data_loader_pku(args)

n_pku_clc = len(np.unique(train_set.train_data['sketch_labels'])) # number of person in training set
n_test_color = len(test_color_set.color_labels) #number of instance in test color set
n_test_sketch = len(test_sketch_set.sketch_labels) #number of instance in test sketch set


pku_train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
pku_test_color_loader = data.DataLoader(test_color_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
pku_test_sketch_loader = data.DataLoader(test_sketch_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)

test_color_labels = test_color_set.color_labels
test_sketch_labels = test_sketch_set.sketch_labels



###############Load Data##############################################


###############Building Model ##############################################
print('==> Building model..')
print('There are {} pids in the train'.format(n_pku_clc))

Embed_net_1 = Baseline(pretrain_choice=None)
Embed_net_2 = Baseline(pretrain_choice=None)
Classify_net = C_net(args.low_dim,n_pku_clc,0.7)
A_net = Attribute_net(dim=args.low_dim, n_att=args.num_att)
D_net = Domain_net(dim=args.low_dim)

trainer = create_trainer(args, Embed_net_1, Embed_net_2, Classify_net, A_net, D_net, n_pku_clc)


# training
best_acc = 0  # best test accuracy
start_epoch = 0 
switch_point = 30

print('==> Start Training...')    
for epoch in range(start_epoch, 101-start_epoch):

    print('==> Preparing Data Loader...')
      
    # training
    if epoch<switch_point:
        trainer.do_train(epoch, pku_train_loader, market_loader, 'softmax')
    else:
        trainer.do_train(epoch, pku_train_loader, market_loader, 'triplet_softmax')
    trainer.adjust_learning_rate(epoch)

    if epoch >= 0 and epoch%5 ==0:
        print ('Test Epoch: {}'.format(epoch))
        # testing
        cmc, mAP = trainer.do_test(epoch, args.test_mode, pku_test_color_loader, pku_test_sketch_loader, n_test_color, n_test_sketch, test_color_labels, test_sketch_labels)

        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))

        # save model
        if cmc[0] > best_acc: # not the real best for sysu-mm01 
            best_acc = cmc[0]
            trainer.save_model(epoch, cmc, mAP, True)

    
    # save model every args.save_epoch epochs    
    if epoch > 0 and epoch%args.save_epoch ==0:
        trainer.save_model(epoch, cmc, mAP, False)
   

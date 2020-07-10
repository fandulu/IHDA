from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.optim as optim
import torch.utils.data as data

from config import Config_zap, Config_sbir
from utils import *
from datasets import *
from net.model import *
from engine.zap_sbir_trainer import *

import random
random.seed(1234)

###############Init Setting#########################################
args = Config_sbir()

checkpoint_path = args.model_path    
    
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

###############Init Setting##########################################


###############Load Data#############################################
zap_args = Config_zap()
zap_loader, _ = make_zap_loader(zap_args)

train_set, test_color_set, test_sketch_set = make_data_loader_sbir(args)

n_sbir_clc = len(np.unique(train_set.train_data['labels'])) 
print(f'number of classes is {n_sbir_clc}')
n_test_color = len(test_color_set.color_labels) 
n_test_sketch = len(test_sketch_set.sketch_labels) 


sbir_train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
sbir_test_color_loader = data.DataLoader(test_color_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)
sbir_test_sketch_loader = data.DataLoader(test_sketch_set, batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers)

test_color_labels = np.array(test_color_set.color_labels)
test_sketch_labels = np.array(test_sketch_set.sketch_labels)
###############Load Data#############################################


###############Building Model #######################################
print('==> Building model..')

Embed_net_1 = Baseline(pretrain_choice=None)
Embed_net_2 = Baseline(pretrain_choice=None)
Classify_net = C_net(args.low_dim,n_sbir_clc, 0.7)
A_net = Attribute_net(dim=args.low_dim, n_att=args.num_att)
D_net = Domain_net(dim=args.low_dim)

trainer = create_trainer(args, Embed_net_1, Embed_net_2, Classify_net, A_net, D_net, n_sbir_clc)
    
# training

best_acc = 0  # best test accuracy
start_epoch = 0 
swith_point = 20

print('==> Start Training...')    
for epoch in range(start_epoch, 61-start_epoch):

    print('==> Preparing Data Loader...')
    
    if epoch < swith_point:
        trainer.do_train(epoch, sbir_train_loader, zap_loader, 'softmax')
    else:
        trainer.do_train(epoch, sbir_train_loader, zap_loader, 'triplet_softmax')
    trainer.adjust_learning_rate(epoch)

    
    if epoch >= 0 and epoch%2 ==0:
        print ('Test Epoch: {}'.format(epoch))
        # testing
        cmc, mAP = trainer.do_test(epoch, sbir_test_color_loader, sbir_test_sketch_loader, n_test_color, n_test_sketch, test_color_labels, test_sketch_labels)

        print('FC: top-1: {:.2%} | top-5: {:.2%} | top-10: {:.2%}| top-20: {:.2%}| mAP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP))

        # save model
        if cmc[0] > best_acc: # not the real best for sysu-mm01 
            best_acc = cmc[0]
            trainer.save_model(epoch, cmc, mAP, True)

    
    # save model every args.save_epoch epochs    
    if epoch > 0 and epoch%args.save_epoch ==0:
        trainer.save_model(epoch, cmc, mAP, False)
   

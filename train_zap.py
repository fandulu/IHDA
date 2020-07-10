from __future__ import print_function
import argparse
import sys
import time 
import torch
import torch.optim as optim

from config import Config_zap
from utils import *
from datasets import *
from net.model import *
from engine.zap_tainer import *


###############Init Setting#########################################
args = Config_zap()
checkpoint_path = args.model_path    

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

###############Init Setting##########################################


###############Load Data#############################################
train_loader, train_num = make_zap_loader(args)
###############Load Data#############################################


###############Building Model #######################################
print('==> Building model..')

Embed_net = Baseline(model_path=args.model_path+'se_resnext50.pth')
Classify_net = C_net(args.low_dim,train_num)
A_net = Attribute_net(dim=args.low_dim, n_att=args.num_att)

trainer = create_trainer(args, Embed_net, Classify_net, A_net, train_num)
    
# training

best_acc = 0  # best test accuracy
start_epoch = 0 
swith_point = 0

print('==> Start Training...')    
for epoch in range(start_epoch, 121-start_epoch):

    print('==> Preparing Data Loader...')
    if epoch == swith_point:
        args.num_instance = 2
        train_loader, train_num = make_zap_loader(args)
    
    if epoch < swith_point:
        trainer.do_train(epoch, train_loader, 'softmax')
    else:
        trainer.do_train(epoch, train_loader, 'triplet_softmax')
    trainer.adjust_learning_rate(epoch)

    
    # save model every args.save_epoch epochs    
    if epoch > 0 and epoch%args.save_epoch ==0:
        trainer.save_model(epoch)
   

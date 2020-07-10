from __future__ import print_function
import argparse
import sys
import time 
import torch

import torch.optim as optim

from config import Config_market
from utils import *
from datasets import *
from net.model import *
from engine.market_tainer import *


###############Init Setting#########################################
args = Config_market()
checkpoint_path = args.model_path    

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

###############Init Setting##########################################



###############Load Data##############################################
market_loader, n_market_clc = make_market_data_loader(args)
###############Load Data##############################################


###############Building Model ##############################################
print('==> Building model..')

Embed_net = Baseline(model_path=args.model_path+'se_resnext50.pth')
Classify_net = C_net(args.low_dim,n_market_clc)
A_net = Attribute_net(dim=args.low_dim, n_att=args.num_att)

trainer = create_trainer(args, Embed_net, Classify_net, A_net, n_market_clc)
    
# training

best_acc = 0  # best test accuracy
start_epoch = 0 
swith_point = 10

print('==> Start Training...')    
for epoch in range(start_epoch, 121-start_epoch):

    print('==> Preparing Data Loader...')
    if epoch == swith_point:
        args.num_instance = 4
        market_loader, n_market_clc = make_market_data_loader(args)
    
    if epoch < swith_point:
        trainer.do_train(epoch, market_loader, 'softmax')
    else:
        trainer.do_train(epoch, market_loader, 'triplet_softmax')
    trainer.adjust_learning_rate(epoch)

    
    # save model every args.save_epoch epochs    
    if epoch > 0 and epoch%args.save_epoch ==0:
        trainer.save_model(epoch)
   

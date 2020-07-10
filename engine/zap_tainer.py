import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math

from utils import *
from net.triplet_loss import make_loss, CrossEntropyLabelSmooth

class create_trainer(object):
    def __init__(self, args, Embed_net, Classify_net, A_net, n_class):
        super(create_trainer, self).__init__()
        
        self.args = args
        self.Embed_net = Embed_net
        self.Classify_net = Classify_net
        self.A_net = A_net
        
        self.triplet_softmax = make_loss('softmax_triplet',n_class)
        self.triplet = make_loss('triplet',n_class)
        self.xent = CrossEntropyLabelSmooth(n_class)
        self.mse = nn.MSELoss()
              
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cudnn.benchmark = True
        
        self.Embed_net.to(self.device)
        self.Classify_net.to(self.device)
        self.A_net.to(self.device)
        
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.Embed_net = nn.DataParallel(self.Embed_net)
            self.Classify_net = nn.DataParallel(self.Classify_net)
            self.A_net = nn.DataParallel(self.A_net)  
            
        if len(self.args.resume)>0:   
            model_path = args.model_path + self.args.resume
            if os.path.isfile(model_path):
                checkpoint = torch.load(model_path)            
                try:
                    self.Embed_net.load_state_dict(checkpoint['Embed_net_1'])
                    self.A_net.load_state_dict(checkpoint['A_net'])
                    print('==> loading checkpoint {}'.format(self.args.resume))
                    self.Classify_net.load_state_dict(checkpoint['Classify_net'])
                except:
                    pass 
            else:
                print('==> no checkpoint found at {}'.format(self.args.resume))
                                               
        
        self.Embed_optimizer = optim.Adam(self.Embed_net.parameters(), lr = 0.3*self.args.lr, betas=(0.5, 0.999))
        self.Classify_optimizer = optim.Adam(self.Classify_net.parameters(), lr = self.args.lr, betas=(0.5, 0.999))
        self.A_optimizer = optim.Adam(self.A_net.parameters(), lr = self.args.lr, betas=(0.5, 0.999))


            
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed """
        if epoch<=10:
            lr = self.args.lr*(epoch+1%11)
        elif epoch>10 and epoch<=30:
            lr = self.args.lr
        elif epoch>30 and epoch<=60:
            lr = self.args.lr*2
        elif epoch>60 and epoch<=80:
            lr = self.args.lr
        else:
            lr = self.args.lr*0.5

            
        print('Base lr: {:.6f}'.format(lr))
        
        self.Embed_optimizer.param_groups[0]['lr'] = 0.3*lr
        self.Classify_optimizer.param_groups[0]['lr'] = lr
        self.A_optimizer.param_groups[0]['lr'] = lr
       
    def reset_grad(self):
        self.Embed_optimizer.zero_grad()
        self.Classify_optimizer.zero_grad()
        self.A_optimizer.zero_grad()
        
    def set_train_mode(self):
        self.Embed_net.train()
        self.Classify_net.train()
        self.A_net.train()
        
    def set_test_mode(self):
        self.Embed_net.eval()
        self.Classify_net.eval()
        self.A_net.eval()
        
    def do_train(self, epoch, zap_loader, loss_mode):
        
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0
        
        self.set_train_mode()
        
        for batch_idx, data in enumerate(zap_loader):
            
            self.reset_grad()

            img_gray, l_zap, att = data       
            img_gray = img_gray.cuda()
            l_zap = l_zap.cuda()
            att = att.type(torch.cuda.FloatTensor)
            att = att.cuda()

            ###zap train#####           
            feat_1, feat_2 = self.Embed_net(img_gray)
            clcs = self.Classify_net(feat_2)
            att_pred = self.A_net(feat_2)

            loss_att = F.binary_cross_entropy(torch.sigmoid(att_pred),att)
            
            if loss_mode == 'softmax':
                loss = self.xent(clcs,l_zap) + loss_att
                           
            elif loss_mode == 'triplet_softmax':   
                loss = self.xent(clcs,l_zap) + self.triplet(feat_1, l_zap) + self.triplet(feat_2, l_zap) + loss_att 
                                         
            loss.backward() 

            self.Embed_optimizer.step()
            self.Classify_optimizer.step()
            self.A_optimizer.step()
            
            self.reset_grad()
            ###################

            if batch_idx%10 ==0:
                print('Epoch: [{}][{}/{}], Loss: {:.2f}'.format(epoch, batch_idx, len(zap_loader), loss.cpu().detach().numpy()))
                print('loss_att:{}'.format(loss_att))
              
    
    def save_model(self,epoch):
        
        state = {
            'Embed_net_1': self.Embed_net.module.state_dict(),
            'Embed_net_2': self.Embed_net.module.state_dict(),
            'Classify_net': self.Classify_net.module.state_dict(),
            'A_net': self.A_net.module.state_dict(),
            'epoch': epoch
        }
        torch.save(state, self.args.model_path + 'zap' + '_epoch_{}.t'.format(epoch))
            



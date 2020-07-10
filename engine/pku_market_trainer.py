import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import math
import random
from itertools import cycle
from net.model import AdversarialLayer

from utils import *
from eval_metrics import eval_pku
from net.triplet_loss import *

class create_trainer(object):
    def __init__(self, args, Embed_net_1, Embed_net_2, Classify_net, A_net, D_net, n_class):
        super(create_trainer, self).__init__()
        
        self.args = args
        self.Embed_net_1 = Embed_net_1
        self.Embed_net_2 = Embed_net_2
        self.Classify_net = Classify_net
        self.A_net = A_net
        self.D_net = D_net
        self.GRL_1 = AdversarialLayer()
        self.GRL_2 = AdversarialLayer()
        self.GRL_3 = AdversarialLayer()
        
        self.triplet_softmax = make_loss('softmax_triplet',n_class)
        self.triplet = make_loss('triplet',n_class)
        self.xent = CrossEntropyLabelSmooth(n_class)
        self.mse = nn.MSELoss()
              
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cudnn.benchmark = True
        
        self.Embed_net_1.to(self.device)
        self.Embed_net_2.to(self.device)
        self.Classify_net.to(self.device)
        self.A_net.to(self.device)
        self.D_net.to(self.device)
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.Embed_net_1 = nn.DataParallel(self.Embed_net_1)
            self.Embed_net_2 = nn.DataParallel(self.Embed_net_2)
            self.Classify_net = nn.DataParallel(self.Classify_net)
            self.A_net = nn.DataParallel(self.A_net)
            self.D_net = nn.DataParallel(self.D_net)
        
        if len(self.args.resume)>0:   
            model_path = self.args.model_path + self.args.resume
            if os.path.isfile(model_path):
                checkpoint = torch.load(model_path)
                self.Embed_net_1.load_state_dict(checkpoint['Embed_net_1'])
                self.Embed_net_2.load_state_dict(checkpoint['Embed_net_2'])   
                self.A_net.load_state_dict(checkpoint['A_net'])
                print('==> loading checkpoint {} Embed_net and A_net'.format(self.args.resume))
                try:
                    self.Classify_net.load_state_dict(checkpoint['Classify_net']) ###
                    self.D_net.load_state_dict(checkpoint['D_net']) ###
                    print('==> loading checkpoint {} Classify_net and D_net'.format(self.args.resume))
                except:
                    pass
            else:
                print('==> no checkpoint found at {}'.format(self.args.resume))
                       
        
        self.Embed_1_optimizer = optim.Adam(self.Embed_net_1.parameters(), lr = 0.2*self.args.lr, betas=(0.5, 0.999))
        self.Embed_2_optimizer = optim.Adam(self.Embed_net_2.parameters(), lr = 0.2*self.args.lr, betas=(0.5, 0.999))
        self.Classify_optimizer = optim.Adam(self.Classify_net.parameters(), lr = self.args.lr, betas=(0.5, 0.999))
        self.A_optimizer = optim.Adam(self.A_net.parameters(), lr = self.args.lr, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D_net.parameters(), lr = self.args.lr, betas=(0.5, 0.999))

        
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed """
        if epoch<=10:
            lr = self.args.lr*((epoch+1)/11)
        elif epoch>10 and epoch<=30:
            lr = self.args.lr
        else:
            lr = self.args.lr*0.8**(epoch/10)
        print('Base lr: {:.6f}'.format(lr))
        
        self.Embed_1_optimizer.param_groups[0]['lr'] =  0.2*lr
        self.Embed_2_optimizer.param_groups[0]['lr'] =  0.2*lr
        self.Classify_optimizer.param_groups[0]['lr'] = lr
        self.A_optimizer.param_groups[0]['lr'] = lr
        self.D_optimizer.param_groups[0]['lr'] = lr
        
    def reset_grad(self):
        self.Embed_1_optimizer.zero_grad()
        self.Embed_2_optimizer.zero_grad()
        self.Classify_optimizer.zero_grad()
        self.A_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        
    def set_train_mode(self):
        self.Embed_net_1.train()
        self.Embed_net_2.train()
        self.Classify_net.train()
        self.A_net.train()
        self.D_net.train()
        
    def set_test_mode(self):
        self.Embed_net_1.eval()
        self.Embed_net_2.eval()
        self.Classify_net.eval()
        self.A_net.eval()
        self.D_net.eval()
        
    def entropy_loss(self, v):
        n, c = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-6))) / (n*np.log2(c))

                
    def do_train(self, epoch, pku_train_loader, market_loader, loss_mode):
        
        train_loss = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        correct = 0
        total = 0
        
        self.set_train_mode()
        
        for batch_idx, data in enumerate(zip(pku_train_loader, market_loader)):
           
            self.reset_grad()
            
            img_pku_gray_1, img_pku_gray_2, img_pku_sketch, l_pku_gray_1, l_pku_gray_2, l_pku_sketch = data[0]
            '''
            img_color = torch.cat((img_color_1,img_color_2),0)
            l_color = torch.cat((l_color_1, l_color_2),0)
            '''
            if bool(random.getrandbits(1)):
                img_pku_gray, l_pku_gray = img_pku_gray_1, l_pku_gray_1
            else:
                img_pku_gray, l_pku_gray = img_pku_gray_2, l_pku_gray_2
               
            img_pku_gray = img_pku_gray.cuda()
            l_pku_gray = (l_pku_gray).cuda()
            
            img_pku_sketch = img_pku_sketch.cuda()
            l_pku_sketch = (l_pku_sketch).cuda()
            
            l_pku = torch.cat((l_pku_gray,l_pku_sketch),0)

            img_mar_gray, _, _, _, att = data[1]
 
            img_mar_gray = img_mar_gray.cuda()
            att = att.type(torch.cuda.FloatTensor)
            att = att.cuda()     
                
            ###################
            feat_mar_gray_1, feat_mar_gray_2 = self.Embed_net_1(img_mar_gray)
            feat_pku_gray_1, feat_pku_gray_2 = self.Embed_net_1(img_pku_gray)
            feat_pku_sketch_1, feat_pku_sketch_2 = self.Embed_net_2(img_pku_sketch)

            feats_pku_1 = torch.cat((feat_pku_gray_1, feat_pku_sketch_1),0)
            feats_pku_2 = torch.cat((feat_pku_gray_2, feat_pku_sketch_2),0)
                    
            ##pid classify####
            pred_clc_pku = self.Classify_net(feats_pku_2)
            ##pid classify####
            
            ##attributes####
            pred_att_mar_gray = self.A_net(feat_mar_gray_2)
            pred_att_pku_gray = self.A_net(feat_pku_gray_2)
            pred_att_pku_sketch = self.A_net(feat_pku_sketch_2)           
            ##attributes####
            
            ##domain####
            pred_d_mar_gray = self.D_net(self.GRL_1(feat_mar_gray_2))
            pred_d_pku_gray = self.D_net(self.GRL_2(feat_pku_gray_2))
            pred_d_pku_sketch = self.D_net(self.GRL_3(feat_pku_sketch_2))
            
            pred_d_all = torch.cat((pred_d_mar_gray,pred_d_pku_gray,pred_d_pku_sketch),0)
            
            d_mark_gray = torch.zeros(pred_d_mar_gray.shape[0])
            d_pku_gray = torch.ones(pred_d_pku_gray.shape[0])
            d_pku_sketch = torch.ones(pred_d_pku_sketch.shape[0])*2
            
            d_all = torch.cat((d_mark_gray,d_pku_gray,d_pku_sketch),0)
            d_all = d_all.long()
            d_all = d_all.cuda() 
            ##domain####
            
            ##losses####
            loss_att_mar_gray = F.binary_cross_entropy(torch.sigmoid(pred_att_mar_gray),att)
            loss_att_pku_gray = self.entropy_loss(torch.sigmoid(pred_att_pku_gray))
            loss_att_pku_sketch = self.entropy_loss(torch.sigmoid(pred_att_pku_sketch))
            loss_att_consist = self.mse(torch.sigmoid(pred_att_pku_gray), torch.sigmoid(pred_att_pku_sketch))
            
            loss_att = (loss_att_mar_gray + loss_att_pku_gray + loss_att_pku_sketch + loss_att_consist)*0.1
            
            loss_domain = -F.cross_entropy(F.softmax(pred_d_all,dim=1),d_all)*0.1 #inverse cross-entropy
            
            if loss_mode == 'softmax':
                loss_clcs = self.xent(pred_clc_pku, l_pku)           
                loss = loss_clcs + loss_att + loss_domain
            elif loss_mode == 'triplet_softmax':  
                loss_triplet_softmax = self.xent(pred_clc_pku, l_pku) + self.triplet(feats_pku_1,l_pku) + self.triplet(feats_pku_2,l_pku)
                loss = loss_triplet_softmax + loss_att + loss_domain
            
            loss.backward() 
            ##losses####

            self.Embed_1_optimizer.step()
            self.Embed_2_optimizer.step()
            self.Classify_optimizer.step()
            self.A_optimizer.step()
            self.D_optimizer.step()
                      
            self.reset_grad()
            ###################
            
            _, predicted = pred_clc_pku.max(1)
            correct += predicted.eq(l_pku).sum().item()
            train_loss.update(loss.item(), l_pku.size(0))
            total += l_pku.size(0)
          
            if batch_idx%10 ==0:
                print('Epoch: [{}][{}/{}], Loss: {train_loss.val:.4f}, Accu: {:.2f}'.format(epoch, batch_idx, len(pku_train_loader),100.*correct/total, train_loss=train_loss))
                print('loss_att_mar_gray:{}'.format(loss_att_mar_gray))
                print('loss_att_pku_gray:{}'.format(loss_att_pku_gray))
                print('loss_att_pku_sketch:{}'.format(loss_att_pku_sketch))
                print('loss_att_consist:{}'.format(loss_att_consist))
                print('loss_domain:{}'.format(loss_domain))
                
        
    def do_test(self, epoch, test_mode, test_color_loader, test_sketch_loader, n_test_color, n_test_sketch, test_color_labels, test_sketch_labels): 
        
        if test_mode == 'color -> sketch':
            nquery = n_test_color
            ngall = n_test_sketch
            #mode_switch = [1,2]
            query_loader = test_color_loader
            gall_loader = test_sketch_loader
            query_label = test_color_labels
            gall_label = test_sketch_labels
        elif test_mode == 'sketch -> color':
            nquery = n_test_sketch
            ngall = n_test_color
            query_loader = test_sketch_loader
            gall_loader = test_color_loader
            query_label = test_sketch_labels
            gall_label = test_color_labels
            
        # switch to evaluation mode
        self.set_test_mode()
        
        print('Do testing')
        print ('Extracting Query Feature...')
        ptr = 0
        query_feat = np.zeros((nquery, self.args.low_dim))
        with torch.no_grad():
            for batch_idx, (img, label ) in enumerate(query_loader):
                batch_num = img.size(0)
                img = img.cuda()
                if self.args.test_mode == 'color -> sketch':
                    _,feat = self.Embed_net_1(img)
                    feat = self.Classify_net(feat)
                elif self.args.test_mode == 'sketch -> color':
                    _,feat = self.Embed_net_2(img)
                    feat = self.Classify_net(feat)
                feat = F.normalize(feat, p=2, dim=1) # normorlize features
                query_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num         
       
        print ('Extracting Gallery Feature...')
        ptr = 0
        gall_feat = np.zeros((ngall, self.args.low_dim))
        
        with torch.no_grad():
            for batch_idx, (img, label ) in enumerate(gall_loader):
                batch_num = img.size(0)
                img = img.cuda()
                if self.args.test_mode == 'color -> sketch':
                    _,feat = self.Embed_net_2(img)
                    feat = self.Classify_net(feat)
                elif self.args.test_mode == 'sketch -> color':
                    _,feat = self.Embed_net_1(img)
                    feat = self.Classify_net(feat)
                feat = F.normalize(feat, p=2, dim=1) # normorlize features
                gall_feat[ptr:ptr+batch_num,: ] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num 

        # compute the similarity
        distmat  = np.matmul(query_feat, np.transpose(gall_feat))

        # evaluation
        cmc, mAP = eval_pku(-distmat, query_label, gall_label)

        return cmc, mAP 
    

    def save_model(self,epoch,cmc, mAP,best=False):
        if best:
            if torch.cuda.device_count() > 1: 
                state = {
                    'Embed_net_1': self.Embed_net_1.module.state_dict(),
                    'Embed_net_2': self.Embed_net_2.module.state_dict(),
                    'Classify_net': self.Classify_net.module.state_dict(),
                    'A_net': self.A_net.module.state_dict(),
                    'D_net': self.D_net.module.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
            else:
                state = {
                    'Embed_net_1': self.Embed_net_1.state_dict(),
                    'Embed_net_2': self.Embed_net_2.state_dict(),
                    'Classify_net': self.Classify_net.state_dict(),
                    'A_net': self.A_net.state_dict(),
                    'D_net': self.D_net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
            torch.save(state, self.args.model_path + 'pku_best.t')
        else:
            if torch.cuda.device_count() > 1:  
                state = {
                    'Embed_net_1': self.Embed_net_1.module.state_dict(),
                    'Embed_net_2': self.Embed_net_2.module.state_dict(),
                    'Classify_net': self.Classify_net.module.state_dict(),
                    'A_net': self.A_net.module.state_dict(),
                    'D_net': self.D_net.module.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
            else:
                state = {
                    'Embed_net_1': self.Embed_net_1.state_dict(),
                    'Embed_net_2': self.Embed_net_2.state_dict(),
                    'Classify_net': self.Classify_net.state_dict(),
                    'A_net': self.A_net.state_dict(),
                    'D_net': self.D_net.state_dict(),
                    'cmc': cmc,
                    'mAP': mAP,
                    'epoch': epoch,
                }
            torch.save(state, self.args.model_path + self.args.dataset + '_epoch_{}.t'.format(epoch))


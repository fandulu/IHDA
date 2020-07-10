import torch
from torch import nn
from .backbones import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
import numpy as np

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0, 0.001)
        nn.init.zeros_(m.bias.data)
            
            
class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, model_path=None, last_stride=1, model_name='se_resnext50', pretrain_choice='imagenet'):
        super(Baseline, self).__init__()
        
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        else:
            print('unsupported backbone! only support resnet50 and se_resnext50, but got {}'.format(model_name))
            
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat_1, feat_2 = self.base(x)
        feat_1 = self.gap(feat_1)  # (b, 2048, 1, 1)
        feat_2 = self.gap(feat_2)  # (b, 2048, 1, 1)
        feat_1 = feat_1.view(feat_1.shape[0], -1).contiguous()  # flatten to (bs, 2048)
        feat_2 = feat_2.view(feat_2.shape[0], -1).contiguous()  # flatten to (bs, 2048)
        return feat_1, feat_2  

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
            
     
    
class C_net(nn.Module):
    def __init__(self, input_dim, class_num, dropout=None):
        super(C_net, self).__init__()
        classifier = []
        #classifier += [nn.BatchNorm1d(input_dim)]
        
        self.bottleneck = nn.BatchNorm1d(input_dim)
        self.bottleneck.bias.requires_grad_(False)
        
        if dropout:
            classifier += [nn.Dropout(p=dropout)] 
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        feat = self.bottleneck(x)
        score = self.classifier(feat)
        if self.training:
            return score
        else:
            return feat
    
    
class Attribute_net(nn.Module):  # attribute net
    def __init__(self, dim=2048, n_att=None):
        super(Attribute_net, self).__init__()
        A_block = []
        
        A_block += [nn.Linear(dim, 128)] 
        A_block += [nn.BatchNorm1d(128)]
        A_block += [nn.LeakyReLU(0.1,inplace=True)]
        
        A_block += [nn.Dropout(p=0.5)]
        
        A_block += [nn.Linear(128, 128)] 
        A_block += [nn.BatchNorm1d(128)]
        A_block += [nn.LeakyReLU(0.1,inplace=True)] 

        A_block += [nn.Dropout(p=0.5)]
        
        A_block += [nn.Linear(128, n_att)] 
        A_block = nn.Sequential(*A_block)
        A_block.apply(weights_init_kaiming)
        self.attribute_net = A_block
        
    def forward(self,x):
        return self.attribute_net(x) 

class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 10000.0
    
    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        return -self.coeff * gradOutput    

    
class Domain_net(nn.Module):  # attribute net
    def __init__(self, dim=2048):
        super(Domain_net, self).__init__()
        D_block = []
        
        D_block += [nn.Linear(dim, 128)] 
        D_block += [nn.BatchNorm1d(128)]
        D_block += [nn.LeakyReLU(0.1,inplace=True)]
        
        D_block += [nn.Linear(128, 128)] 
        D_block += [nn.BatchNorm1d(128)]
        D_block += [nn.LeakyReLU(0.1,inplace=True)] 

        D_block += [nn.Linear(128, 3)] 
        D_block = nn.Sequential(*D_block)
        D_block.apply(weights_init_kaiming)
        self.domain_net = D_block
        
    def forward(self,x):
        return self.domain_net(x) 
            
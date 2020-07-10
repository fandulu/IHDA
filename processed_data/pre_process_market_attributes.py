import sys
sys.path.extend(['../'])

import os
import glob
import numpy as np
from PIL import Image, ImageChops
from scipy.io import loadmat
import pickle
from config import Config_market


def main():
    Cfg = Config_market()
    attribute = loadmat(Cfg.raw_att)
    
    mdata = attribute['market_attribute']['train'][0,0]
    mtype = mdata.dtype
    ndata = {n: mdata[n][0,0] for n in mtype.names}
    n_pid = len(ndata['image_index'][0]) # number of pid
    
    att_table = {}
    for i, img_id in enumerate(ndata['image_index'][0]):
        att = np.array([ndata['gender'][0][i],
                        ndata['hair'][0][i],
                        ndata['up'][0][i],
                        ndata['down'][0][i],
                        ndata['clothes'][0][i],
                        ndata['hat'][0][i],
                        ndata['backpack'][0][i],
                        np.max([ndata['handbag'][0][i],ndata['bag'][0][i]])
                       ])
        att[att==2] = 0
        att_table[img_id[0]] = att

    f = open('market/market_attributes.pkl', 'wb')
    pickle.dump(att_table,f)
    f.close()

    print('Market attributes are generated!')
    
if __name__== "__main__":
    main()
    
'''
ndata['upblack'][0][i],
ndata['upwhite'][0][i],
ndata['upred'][0][i],
ndata['uppurple'][0][i],
ndata['upyellow'][0][i],
ndata['upgray'][0][i],
ndata['upblue'][0][i],
ndata['upgreen'][0][i],

ndata['downblack'][0][i],
ndata['downwhite'][0][i],
ndata['downpink'][0][i],
ndata['downpurple'][0][i],
ndata['downyellow'][0][i],
ndata['downgray'][0][i],
ndata['downblue'][0][i],
ndata['downgreen'][0][i],
ndata['downbrown'][0][i] 
'''
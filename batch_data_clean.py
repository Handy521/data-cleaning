#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:15:02 2019

@author: shinong
"""

import os
import shutil
# read codes and labels from file 
import csv 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
labels_dir= '/home/shinong/Desktop/labels/'
codes_dir= '/home/shinong/Desktop/codes/'

labels_all=sorted(os.listdir(labels_dir))
codes_all=sorted(os.listdir(codes_dir))
def get_cossimi(x,y):
    myx = np.array(x) 
    myy = np.array(y) 
    cos1 = np.sum(myx*myy) 
    cos21 = np.sqrt(sum(myx*myx)) 
    cos22 = np.sqrt(sum(myy*myy)) 
    return cos1/float(cos21*cos22) 

for ii,label in enumerate(labels_all):
    label_f=labels_dir+label
    with open(label_f) as f: 
        reader = csv.reader(f, delimiter='\n') 
        labels = np.array([each for each in reader if len(each) > 0]).squeeze() 
    
    code_f=codes_dir+codes_all[ii] 
    with open(code_f) as f2:
        codes = np.fromfile(f2, dtype=np.float32) 
        codes = codes.reshape((len(labels), -1))

    mean_code=np.mean(codes,axis=0) # axis=0，计算每一列的均值        
    cos_values=[] 
    without_values={}
    for i in range(len(codes)):    
        cos_value=get_cossimi(codes[i,:],mean_code)
        cos_values.append(cos_value)
        if cos_value<0.48:
            without_values[i]=cos_value
    label_dict={}
    for i in range(len(labels)):
        label_dict[i]=labels[i]
    
    for k1,v1 in without_values.items():#new dir save image
        for k2,v2 in label_dict.items():
            if k1==k2:
                path=v2
                if not os.path.exists(str(ii)):
                    os.mkdir(str(ii))
                if os.path.isfile(path):
                    shutil.move(path,str(ii))
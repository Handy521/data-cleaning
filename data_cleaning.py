#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:42:31 2019

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
with open('labels1') as f: 
    reader = csv.reader(f, delimiter='\n') 
    labels = np.array([each for each in reader if len(each) > 0]).squeeze() 
with open('codes1') as f: 
    codes = np.fromfile(f, dtype=np.float32) 
    codes = codes.reshape((len(labels), -1))
mean_code=np.mean(codes,axis=0) # axis=0，计算每一列的均值
#mean_code2=np.mean(codes,axis=1) # 计算每一行的均值
data_clean_dir="clean_data1"
if not os.path.exists(data_clean_dir):
    os.mkdir(data_clean_dir)
def get_cossimi(x,y):
    myx = np.array(x) 
    myy = np.array(y) 
    cos1 = np.sum(myx*myy) 
    cos21 = np.sqrt(sum(myx*myx)) 
    cos22 = np.sqrt(sum(myy*myy)) 
    return cos1/float(cos21*cos22)
cos_values=[] 
without_values={}
for i in range(len(codes)):
    
    cos_value=get_cossimi(codes[i,:],mean_code.T)
    cos_values.append(cos_value)
    if cos_value<0.4:
#        print(i)
        without_values[i]=cos_value
a=np.ones(1356,np.float32)*0.42
plt.plot(a)
plt.plot(cos_values,'ro')
plt.ylabel('some numbers')
plt.show()   

#label_dict={}
#for i in range(len(labels)):
#    label_dict[i]=labels[i]
#
#for k1,v1 in without_values.items():#new dir save image
#    for k2,v2 in label_dict.items():
#        if k1==k2:
#            path=v2
##        if not os.path.exists(v):
##            os.mkdir(v)
##        path=os.path.join(image_dir,v,k)
#            if os.path.isfile(path):
#                shutil.move(path,data_clean_dir)
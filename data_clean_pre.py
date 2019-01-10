#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:32:35 2019

@author: shinong
"""

import os

import numpy as np
import tensorflow as tf

import vgg16
import utils
import cv2
data_dir = '/media/shinong/study/try/all_image/一年蓬/'
#将图像批量batches通过VGG模型，将输出作为新的输入：
# Set the batch size higher if you can fit in in your GPU memory
batch_size = 10
codes_list = []
labels = []
batch = []
codes = None

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)

    print("Starting  images")

    files = os.listdir(data_dir)
    for ii, file in enumerate(files, 1):
        # Add images to the current batch
        # utils.load_image crops the input images for us, from the center
        img = utils.load_image(os.path.join(data_dir, file))
        batch.append(img.reshape((1, 224, 224, 3)))            
        labels.append(os.path.join(data_dir, file))
        
        # Running the batch through the network to get the codes
        if ii % batch_size == 0 or ii == len(files):
            images = np.concatenate(batch)

            feed_dict = {input_: images}
            codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
            
            # Here I'm building an array of the codes
            if codes is None:
                codes = codes_batch
            else:
                codes = np.concatenate((codes, codes_batch))
            
            # Reset to start building the next batch
            batch = []
            print('{} images processed'.format(ii))
with open('codes1', 'w') as f:
    codes.tofile(f)
    
import csv
with open('labels1', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)


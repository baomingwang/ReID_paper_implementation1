# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:38:25 2016

@author: lenovo
"""

from __future__ import division
import numpy as np
import os
import re
import scipy.ndimage as ndi
from PIL import Image
import itertools
import random
import pickle
import pdb
#dataset_path = 'D:\Market1501\\dataset\\boundingboxtrain'
dataset_path='/home/ubuntu/dataset/market1501/boundingboxtrain'

def prepare_dataset(image_list, sample_number=2, np_ratio=5):    
    image_cate = [c[0:4] for c in image_list]
    image_set = list(set(image_cate))
    image_id_pool = []
    image_pool_array = []
    positive_list = []
    negative_list = []
    pattern = re.compile('.*jpg$')
    p_nb = 0

    for pp, image_id in enumerate(image_set):
        image_pool_array=[]
        for image1 in image_list:
            if image1[0:4] == image_id:
                image_id_pool.append(image1)
        for image2 in image_id_pool:
            if pattern.match(image2):           
                image_pool_array.append(image_to_array(dataset_path+'/'+image2))
        for i in range(len(image_pool_array)):
            for _ in range(sample_number):
                x = random_shift(image_pool_array[i], 0.05, 0.05)
                image_pool_array.append(x)
        positive_list.append(list(itertools.combinations(image_pool_array,2)))
        #p_nb = p_nb + (len(image_id_pool)*1)*(len(image_id_pool)*1-1) / 2
        image_id_pool=[]
        print pp
    #pdb.set_trace()
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    positive_tuple = flatten(positive_list)
    positive = [list(c) for c in positive_tuple]
    [c.append(1) for c in positive]
    random.shuffle(image_list)
    for j, image_a in enumerate(image_list):
        for image_b in image_list[j:]:
            if image_a[0:4] != image_b[0:4]:
                if pattern.match(image_a) and pattern.match(image_b):                
                    negative_list.append([image_to_array(dataset_path+'/'+image_a), image_to_array(dataset_path+'/'+image_b)])
                    #negative_list.append([image_a, image_b])
                    if len(negative_list) >= len(positive)*np_ratio/2:
                        break
        else:
            print j
            continue
        break
    [c.append(0) for c in negative_list]
    negative_list.extend(positive)
    print 'complete prepare data'   
    return negative_list
    
def image_to_array(image_dir):
    return np.array(Image.open(image_dir))    

def random_shift(x, wrg, hrg, row_index=0, col_index=1, channel_index=2,
                 fill_mode='nearest', cval=0.):
    h, w = x.shape[row_index], x.shape[col_index]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x
    
def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def get_image_path_list(dataset_dir):
    '''
    get a list containing all the paths of images in the trainset
    ---------------------------------------------------------------------------
    INPUT:
        train_or_test: select dataset to train or test
    OUTPUT:
        a list with all the images' paths
    ---------------------------------------------------------------------------
    '''
    folder_path = dataset_dir
    assert os.path.exists(folder_path)
    assert os.path.isdir(folder_path)
    print 'already get all the image path.'
    return os.listdir(folder_path)
    
def save_to_file(obj, filename):
    print 'begin storing data'    
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print 'complete storing data'
    
    
if __name__ == '__main__':
    image_list = get_image_path_list(dataset_path)
    dataset = prepare_dataset(image_list, 0)
    save_to_file(dataset, 'data.pkl')

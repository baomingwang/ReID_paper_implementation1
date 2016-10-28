# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:14:35 2016
@author: dingning
"""
import numpy as np

np.random.seed(1217)

from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
from input_generator import DataSet
from input_generator import load_from_file
import pickle
import pdb

K._IMAGE_DIM_ORDERING = 'tf'
the_filename = 'data_by_path.pkl'
dir_path = '/home/ubuntu/dataset/market1501/boundingboxtrain'

def concat_iterat(input_tensor):
    #pdb.set_trace()
    input_expand = K.expand_dims(K.expand_dims(input_tensor, -2), -2)
    #pdb.set_trace()
    x_axis = []
    y_axis = []
    for x_i in range(5):
        for y_i in range(5):
            y_axis.append(input_expand)
        x_axis.append(K.concatenate(y_axis, axis=2))
        y_axis = []
    return K.concatenate(x_axis, axis=1)

def cross_input_both(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[1]
    y_length = K.int_shape(tensor_left)[2]
    cross_y_left = []
    cross_x_left = []
    cross_y_right = []
    cross_x_right = []
    scalar = K.ones([5,5])
    tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y_left.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
            cross_y_right.append(tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_left_padding[:,i_x,i_y,:]))
        cross_x_left.append(K.concatenate(cross_y_left, axis=2))
        cross_x_right.append(K.concatenate(cross_y_right, axis=2))
        cross_y_left = []
        cross_y_right = []
    cross_out_left = K.concatenate(cross_x_left,axis=1)
    cross_out_right = K.concatenate(cross_x_right,axis=1)
    cross_out = K.concatenate([cross_out_left, cross_out_right], axis=3)
    return K.abs(cross_out)

def cross_input_1(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[1]
    y_length = K.int_shape(tensor_left)[2]
    cross_y = []
    cross_x = []
    tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y.append(tensor_left_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:] 
                         - concat_iterat(tensor_right_padding[:,i_x,i_y,:]))
        cross_x.append(K.concatenate(cross_y,axis=2))
        cross_y = []
    cross_out = K.concatenate(cross_x,axis=1)
    return K.abs(cross_out)

def cross_input_2(X):
    tensor_left = X[0]
    tensor_right = X[1]
    x_length = K.int_shape(tensor_left)[1]
    y_length = K.int_shape(tensor_left)[2]
    cross_y = []
    cross_x = []
    tensor_left_padding = K.spatial_2d_padding(tensor_left,padding=(2,2))
    tensor_right_padding = K.spatial_2d_padding(tensor_right,padding=(2,2))
    for i_x in range(2, x_length + 2):
        for i_y in range(2, y_length + 2):
            cross_y.append(concat_iterat(tensor_right_padding[:,i_x,i_y,:]) 
                         - tensor_right_padding[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
        cross_x.append(K.concatenate(cross_y,axis=2))
        cross_y = []
    cross_out = K.concatenate(cross_x,axis=1)
    return K.abs(cross_out)

def cross_input_shape_both(input_shapes):
    input_shape = input_shapes[0]
    return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3]*2)

def cross_input_shape_single(input_shapes):
    input_shape = input_shapes[0]
    return (input_shape[0],input_shape[1] * 5,input_shape[2] * 5,input_shape[3])    
    
a1 = Input(shape=(128,64,3))
b1 = Input(shape=(128,64,3))
share = Convolution2D(20,5,5,dim_ordering='tf')
a2 = share(a1)
b2 = share(b1)
a3 = Activation('relu')(a2)
b3 = Activation('relu')(b2)
a4 = MaxPooling2D(dim_ordering='tf')(a3)
b4 = MaxPooling2D(dim_ordering='tf')(b3)
share2 = Convolution2D(25,5,5,dim_ordering='tf')
a5 = share2(a4)
b5 = share2(b4)
a6 = Activation('relu')(a5)
b6 = Activation('relu')(b5)
a7 = MaxPooling2D(dim_ordering='tf')(a6)
b7 = MaxPooling2D(dim_ordering='tf')(b6)
a8 = merge([a7,b7],mode=cross_input_1,output_shape=cross_input_shape_single)
b8 = merge([a7,b7],mode=cross_input_2,output_shape=cross_input_shape_single)
a9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu')(a8)
b9 = Convolution2D(25,5,5, subsample=(5,5), dim_ordering='tf',activation='relu')(b8)
a10 = MaxPooling2D((2,2),dim_ordering='tf')(a9)
b10 = MaxPooling2D((2,2),dim_ordering='tf')(b9)
a11 = Convolution2D(25,3,3,dim_ordering='tf',activation='relu')(a10)
b11 = Convolution2D(25,3,3,dim_ordering='tf',activation='relu')(b10)
a12 = MaxPooling2D((2,2),dim_ordering='tf')(a11)
b12 = MaxPooling2D((2,2),dim_ordering='tf')(b11)
c1 = merge([a12, b12], merge_mode='concat', concat_axis=-1)
c2 = Flatten()(c1)
c3 = Dense(500,activation='relu')(c2)
c4 = Dense(1,activation='softmax')(c3)

model = Model(input=[a1,b1],output=c4)
model.summary()    

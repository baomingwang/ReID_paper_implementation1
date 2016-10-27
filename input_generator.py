# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:45:46 2016

@author: lenovo
"""

import numpy as np
import pickle
import random
from prepare_dataset import image_to_array
from keras.utils import np_utils
import pdb

dir_path = '/home/ubuntu/dataset/market1501/boundingboxtrain'
the_filename = 'data_by_path.pkl'

def load_from_file(filename):
    with open(filename, 'rb') as f:
        var = pickle.load(f)
    return var
    
class DataSet(object):

    def __init__(self, data, batch_size=128, random_shuffle=True):
        self._data = data
        self.batch_size = batch_size
        if random_shuffle:
            random.shuffle(self._data)
        self._index_in_epoch = 0
        self._data_num = len(self._data)
        self._epochs_completed = 0
        self._num_batch = self._data_num // self.batch_size
        
    @property
    def data(self):
        return self._data

    @property
    def data_num(self):
        return self._data_num

    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    def next_batch(self):
        while(1):            
            start = self._index_in_epoch
            self._index_in_epoch += self.batch_size
            if self._index_in_epoch > self._data_num:      
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                random.shuffle(self._data)
                # Start next epoch
                start = 0
                self._index_in_epoch = self.batch_size
                assert self.batch_size <= self._num_examples
            end = self._index_in_epoch
            in_path1 = [c[0] for c in self._data[start:end]]
            in_path2 = [c[1] for c in self._data[start:end]]
            in_put1 = np.array([image_to_array(dir_path+'/'+c) for c in in_path1])
            in_put2 = np.array([image_to_array(dir_path+'/'+c) for c in in_path2])
            label = np_utils.to_categorical(np.array([c[2] for c in self._data[start:end]]))
            yield ([in_put1, in_put2], label)
            
if __name__ == '__main__':
    dataset = load_from_file(the_filename)
    market = DataSet(dataset, 10)
    a, b = market.next_batch()
    pdb.set_trace()

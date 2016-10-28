import numpy as np
np.random.seed(1234)
from keras.models import Model
from keras.layers import Input,Dense,Convolution2D,Activation,MaxPooling2D,Flatten,merge
from sgd_paper import SGD_paper
from keras.utils import np_utils
import pickle
from keras import backend as K

a1 = Input(shape=(128,64,3))
b1 = Input(shape=(128,64,3))
share = Convolution2D(20,5,5,dim_ordering='tf', activity_regularizer=l2(l=0.0005))
a2 = share(a1)
b2 = share(b1)
a3 = Activation('relu')(a2)
b3 = Activation('relu')(b2)
a4 = MaxPooling2D(dim_ordering='tf')(a3)
b4 = MaxPooling2D(dim_ordering='tf')(b3)
share2 = Convolution2D(25,5,5,dim_ordering='tf', activity_regularizer=l2(l=0.0005))
a5 = share2(a4)
b5 = share2(b4)
a6 = Activation('relu')(a5)
b6 = Activation('relu')(b5)
a7 = MaxPooling2D(dim_ordering='tf')(a6)
b7 = MaxPooling2D(dim_ordering='tf')(b6)
c1 = merge([a7,b7], mode=cross_input, output_shape=cross_input_shape)
c2 = Convolution2D(25,5,5,dim_ordering='tf',activation='relu', activity_regularizer=l2(l=0.0005))(c1)
c3 = MaxPooling2D((2,2),dim_ordering='tf')(c2)
c4 = Convolution2D(25,3,3,dim_ordering='tf',activation='relu', activity_regularizer=l2(l=0.0005))(c3)
c5 = MaxPooling2D((2,2),dim_ordering='tf')(c4)
c6 = Flatten()(c5)
c7 = Dense(10,activation='relu', activity_regularizer=l2(l=0.0005))(c6)
c8 = Dense(2,activation='softmax')(c7)

model = Model(input=[a1,b1],output=c8)
model.summary()

sgd = SGD_paper(lr=0.01, momentum=0.9)

model.compile(optimizer='RMSprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

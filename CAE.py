from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Dropout
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
import numpy as np

np.random.seed(5)
from tensorflow import set_random_seed
set_random_seed(5)


def CAE(input_shape=(80, 80, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    
    
    model.add(Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='Conv1', input_shape=input_shape))
    

    model.add(Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='Conv2'))
    

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='Conv3'))

    model.add(Flatten(name='Flatten2D'))
    model.add(Dense(units=filters[3], name='Embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu',name='ShapeBeforeEmbedding'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2] ), name='1Dto2D'))
    
   
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='Deconv3'))

    model.add(Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='Deconv2'))
   

    model.add(Conv2DTranspose(input_shape[2], 3, strides=2, padding='same',activation='sigmoid', name='Deconv1'))
    
    model.summary()
    return model

import tensorflow as tf
import numpy as np
from keras.initializers import glorot_uniform
import keras.backend as K

def get_lineal():
    """
    Input shape= (32,32,3)
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    model.add(tf.keras.layers.Dense(units=4, activation = 'sigmoid'))
    return model

def get_lenet():

    """
    Input shape= (32,32,3)
    """

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=6, kernel_size=(5,5), activation='tanh',
            padding='same', input_shape=(32, 32, 3)
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(16, (5,5), activation='tanh'))
    model.add(tf.keras.layers.AveragePooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(120, (5,5), activation='tanh'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(84, activation='tanh'))
    model.add(tf.keras.layers.Dense(units=4, activation = 'sigmoid'))
    return model

def get_alexnet():
    
    """
    Input shape= (224,224,3)
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding= 'valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.Dense(units=4, activation = 'sigmoid'))
    return(model)
    
def getVGG():

    """
    Input shape= (224,224,3)
    """

    model = tf.keras.Sequential()
    # capa 1 good
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3),strides=1,padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 2
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 3
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 4
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #capa 5
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding="same"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=2))
    #flatten
    model.add(tf.keras.layers.Flatten())
    #dense 1 
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 2
    model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
    #dense 3 
    model.add(tf.keras.layers.Dense(units=1000,activation="relu"))
    #capa final
    model.add(tf.keras.layers.Dense(units=4, activation = 'sigmoid'))
    return model

def get_VGG16():

    """
    Input shape= (224,224,3)
    """

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=(224, 224, 3)
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3,3), activation='relu',
            padding='same'
        )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(
        tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3,3), activation='relu',
            padding='same'
            )
    )
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dense(units=4, activation = 'sigmoid'))
    return model


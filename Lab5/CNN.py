import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
import matplotlib

from utilities import get_data, plot_res, plot_heatmap

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.Session(config=config)

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    plt.show()


## CNN
train_x,train_y,test_x,test_y = get_data(size=150)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model = Sequential()
model.add(Conv2D(64, (3,3), padding = 'same', input_shape = (150,150,1)))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3,3), padding = 'same'))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3,3), padding = 'same'))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3,3), padding = 'same'))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, (3,3), padding = 'same'))
model.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
model.add(LeakyReLU(alpha=0.1))

model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dense(15, activation = 'softmax'))

model.compile(optimizer = Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_x,train_y,epochs=20,batch_size=32, validation_data = (test_x,test_y))

model.evaluate(test_x,test_y)

model.save('model_CNN.h5')

plot_history(history)

#model = load_model('model_CNN.h5')

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./result/model_CNN.png',show_shapes=True, show_layer_names=True)

pred = model.predict(test_x)

pred_y = [np.argmax(x) for x in pred]
true_y = [np.argmax(x) for x in test_y]

plot_heatmap(true_y, pred_y, './result/heatmap_CNN')
plot_res(true_y, pred_y, './result/res_CNN')


## VGG16 place 365
from vgg16_places_365 import VGG16_Places365

train_x,train_y,test_x,test_y = get_data(gray=False,size=256)
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

vgg = VGG16_Places365(weights='places', include_top=False, input_shape=(256,256,3))
model_vgg = Sequential()
model_vgg.add(vgg)
model_vgg.add(Flatten())
model_vgg.add(Dense(4096,activation='relu'))
model_vgg.add(Dense(15, activation='softmax'))
vgg.trainable=False

model_vgg.compile(optimizer = Adam(lr=2e-5), loss = "categorical_crossentropy", metrics=["accuracy"])

history_vgg = model_vgg.fit(train_x,train_y,epochs=20,batch_size=32,validation_data=(test_x,test_y))

model_vgg.evaluate(test_x,test_y)

model_vgg.save('model_vgg.h5')

plot_history(history_vgg)

#model_vgg = load_model('model_vgg.h5')

plot_model(model_vgg, to_file='./result/model_VGG.png',show_shapes=True, show_layer_names=True)

pred = model_vgg.predict(test_x)

pred_y = [np.argmax(x) for x in pred]
true_y = [np.argmax(x) for x in test_y]

plot_heatmap(true_y, pred_y, './result/heatmap_VGG')
plot_res(true_y, pred_y, './result/res_VGG')





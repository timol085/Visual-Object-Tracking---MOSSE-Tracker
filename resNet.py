import cv2
from matplotlib import gridspec, pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, utils
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation,MaxPooling2D,Dropout, Add,ZeroPadding2D,MaxPool2D,AveragePooling2D
from keras.utils import plot_model

from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input

def getModel():
    model = ResNet50(weights = 'imagenet')
    return model
    

def resNet(img, model):
    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize image to 224x224 size
    image = cv2.resize(orig, (224, 224)).reshape(-1, 224, 224, 3)
    # We need to preprocess imageto fulfill ResNet50 requirements
    image = preprocess_input(image)
    #model = ResNet50(weights = 'imagenet', include_top = False, classes=3)
    preds = model.predict(image)
    label = tf.keras.applications.resnet.decode_predictions(preds, top=1)[0][0][1]

    
    
    '''n_features = features.shape[-1]

    fig = plt.figure(figsize = (17, 8))
    gs = gridspec.GridSpec(1, 2, figure = fig)
    sub_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(orig)
    
    for i in range(3):
        for j in range(3):
            ax2 = fig.add_subplot(sub_gs[i, j])
            plt.axis('off')        
            plt.imshow(features[0, :, :, np.random.randint(n_features)], cmap = 'gray')
    plt.show()'''
    return preds


'''print(features.shape)

n_features = features.shape[-1]

fig = plt.figure(figsize = (17, 8))
gs = gridspec.GridSpec(1, 2, figure = fig)
sub_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])

ax1 = fig.add_subplot(gs[0])
ax1.imshow(orig)

for i in range(3):
    for j in range(3):
        ax2 = fig.add_subplot(sub_gs[i, j])
        plt.axis('off')        
        plt.imshow(features[0, :, :, np.random.randint(n_features)], cmap = 'gray') 
        
plt.show()'''
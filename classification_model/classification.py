from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from sklearn import datasets 

from pymatreader import read_mat

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from utils import angle_error, RotNetDataGenerator
from tensorflow.keras import datasets, layers, models


data = read_mat("imagelabels.mat")

def unique(list1):
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    
    print("unique_list size = ",len(unique_list))
    return len(unique_list) + 1

import glob
train_filenames = []
for img in glob.glob("./datasets/102flowers/jpg/*.jpg"):
    train_filenames.append(img)

y = data['labels']
# number of classes
nb_classes = unique(y)

x_data = []
# load the image
from PIL import Image
import numpy as np
from numpy import asarray
for img_name in train_filenames:
    #image = Image.open(img_name)
    load_img_rz = np.array(Image.open(img).resize((224,224)))
    x_data.append(load_img_rz)

# converting list to array
x_data = np.array(x_data)


#split the data in train _test val set
x_train = x_data[:6000]
print("X-train shape",x_train.shape)
y_train = y[:6000]

x_test = x_data[6000:]
print("X-train shape",x_train.shape)
y_test = y[6000:]


# input image shape
input_shape = (224, 224, 3)


#Retnet Model
model = load_model('./Rotnet_model/rotnet_street_view_resnet50.hdf5')
model.layers.pop()
model.layers.pop()
model.add(layers.MaxPooling2D())
model.add(layers.Dense(nb_classes, activation='softmax'))
model.summary()


# model compilation
model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model fit
tensorboard_callback = TensorBoard(log_dir="./logs")
model.fit(x_train,y_train, batch_size=32, epochs = 10, callbacks=[tensorboard_callback])

#Evaluate
model.evaluate(x_test,y_test)

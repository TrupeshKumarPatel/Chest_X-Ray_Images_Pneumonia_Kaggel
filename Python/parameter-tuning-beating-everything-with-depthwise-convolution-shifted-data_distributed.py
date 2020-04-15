#!/usr/bin/env python
# coding: utf-8

# ## Source:
# [1] https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution/log
#
# [2] https://data.mendeley.com/datasets/rscbjbr9sj/2

# In[1]:


# get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.clear_all_output();')


# In[2]:


# get_ipython().run_line_magic('reset', '-f')
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
#
# get_ipython().run_line_magic('who', '')


# In[3]:


import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
# from keras import backend as K
# color = sns.color_palette()
# get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

import sys
import datetime
import time

print("Python version: ", sys.version)
print("Version info.: ", sys.version_info)
print("TensorFlow version: ", tf.__version__)
print("TensorFlow.Keras version : ", tf.keras.__version__)


# In[4]:

# Turn interactive plotting off
plt.ioff()
# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(111)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.gpu_options.visible_device_list='0,1,2,3'

# Set the random seed in tensorflow at graph level
tf.compat.v1.set_random_seed(111)

# Set the session in tensorflow
sess = tf.compat.v1.Session(config=config)

# Set the session in keras
tf.compat.v1.keras.backend.set_session(sess)

tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()

print("#--#--"*10)
print('Number of devices: {}\n'.format(strategy.num_replicas_in_sync))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Make the augmentation sequence deterministic
# aug.seed(111)


# ## Load Data

# In[5]:
FileTime = str(datetime.datetime.now().strftime("%m-%d-%Y-%I-%M"))
print("#--#--"*10, "\nFile time: ", FileTime, "\n\n")

# Define path to the data directory
data_dir = Path('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Data/new_chest_xray/')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'

# Path to validation directory
val_dir = data_dir / 'val'

# Path to test directory
test_dir = data_dir / 'test'

# In[6]:


# Get the path to the normal and pneumonia sub-directories
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in sorted(normal_cases):
    train_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in sorted(pneumonia_cases):
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# Shuffle the data
train_data = train_data.sample(random_state=111, frac=1.).reset_index(drop=True)
# train_data = train_data.sample(frac=1.).reset_index(drop=True)

# How the dataframe looks like?
print("#--#--"*10,"\n\n",train_data.head())


# In[7]:


# Get the counts for each class
cases_count = train_data['label'].value_counts()
print(cases_count)

# Plot the results
# plt.figure(figsize=(10,8))
# sns.barplot(x=cases_count.index, y= cases_count.values)
# plt.title('Number of cases', fontsize=14)
# plt.xlabel('Case type', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Pneumonia(1)'])
# plt.show()


# In[8]:


# Get few samples for both the classes
pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()

# Concat the data in a single list and del the above two list
samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

# Plot the data
# f, ax = plt.subplots(2,5, figsize=(30,10))
# for i in range(10):
#     img = imread(samples[i])
#     ax[i//5, i%5].imshow(img, cmap='gray')
#     if i<5:
#         ax[i//5, i%5].set_title("Pneumonia")
#     else:
#         ax[i//5, i%5].set_title("Normal")
#     ax[i//5, i%5].axis('off')
#     ax[i//5, i%5].set_aspect('auto')
# plt.show()


# In[9]:


# Get the path to the sub-directories
normal_cases_dir = val_dir / 'NORMAL'
pneumonia_cases_dir = val_dir / 'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

# List that are going to contain validation images data and the corresponding labels
# An empty list. We will insert the data into this list in (img_path, label) format
valid_data = []


# Go through all the normal cases. The label for these cases will be 0
for img in sorted(normal_cases):
    valid_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in sorted(pneumonia_cases):
    valid_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list
valid_data = pd.DataFrame(valid_data, columns=['image', 'label'],index=None)

# Shuffle the data
valid_data = valid_data.sample(random_state=111, frac=1.).reset_index(drop=True)
# train_data = train_data.sample(frac=1.).reset_index(drop=True)

# How the dataframe looks like?
print("#--#--"*10,"\n\n",valid_data.head())


# In[10]:


# Augmentation sequence
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness


# In[11]:


def data_gen(data, batch_size, augment=False):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # Initialize a counter
    i =0
    while True:
#         np.random.shuffle(indices)
        # Get the next batch
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
#             print("IMG-NAME: ", str(img_name).split("/")[-1])
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))

            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])

            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label


#             count+=1
            # generating more samples of the undersampled class
            if augment:
                if label==0 and count < batch_size-2:
                    aug_img1 = seq.augment_image(img)
                    aug_img2 = seq.augment_image(img)
                    aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                    aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                    aug_img1 = aug_img1.astype(np.float32)/255.
                    aug_img2 = aug_img2.astype(np.float32)/255.

                    batch_data[count+1] = aug_img1
                    batch_labels[count+1] = encoded_label
                    batch_data[count+2] = aug_img2
                    batch_labels[count+2] = encoded_label
                    count +=2

                else:
                    count+=1
            else:
                count+=1

            if count==batch_size:
                break

        i+=1
        yield batch_data, batch_labels

        if i>=steps:
            i=0

# a =  data_gen(data=train_data, batch_size=16)
# for x, y in a:
#     for img, lab in zip(x,y):
#         plt.title(lab)
#         plt.imshow(img, cmap='gray')
#         plt.show()
#     break
# In[12]:


def build_model():
    input_img = Input(shape=(224,224,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)

    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)

    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model


# In[13]:


with strategy.scope():
    model =  build_model()
    opt = Adam(lr=1e-4, amsgrad=True, clipnorm=1.)
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)


# In[14]:


print(model.summary())


# In[15]:


# Open the VGG16 weight file
f = h5py.File('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Data/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
print(model.summary())


# In[16]:


# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}\n'.format(epoch + 1, model.optimizer.lr.numpy()))


# In[17]:


callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, verbose=1, mode='max'),
    ModelCheckpoint(filepath='/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/best_model_todate_python/'+FileTime+'/', monitor='val_loss', verbose=1,
                    save_best_only=True, save_weights_only=True),
    PrintLR()
]

# es = EarlyStopping(patience=5)
# chkpt = ModelCheckpoint(filepath='./best_model_todate', save_best_only=True, save_weights_only=True)
# In[18]:


# batch_size = 16
nb_epochs = 20
BUFFER_SIZE = len(train_data)

BATCH_SIZE_PER_REPLICA = 80
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
batch_size = BATCH_SIZE

print("#--#--"*10,  "\n BATCH_SIZE_PER_REPLICA = ", BATCH_SIZE_PER_REPLICA,
                    "\n BATCH_SIZE = ", BATCH_SIZE,
                    "\n EPOCHS = ", nb_epochs)

# Get a train data generator
train_data_gen = data_gen(data=train_data, batch_size=batch_size)

# Get a valid data generator
val_data_gen = data_gen(data=valid_data, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

# Define the number of validation steps
nb_valid_steps = valid_data.shape[0]//batch_size

print("#--#--"*10,"\n\nNumber of training and validation steps: {} and {}".format(nb_train_steps, nb_valid_steps))#len(valid_data)

# train_dataset = tf.data.Dataset.from_generator(lambda: map(tuple, train_data_gen), (tf.float32, tf.string)).shuffle(BUFFER_SIZE).batch(batch_size)
# val_dataset = tf.data.Dataset.from_generator(lambda: map(tuple, val_data_gen), (tf.float32, tf.string)).batch(batch_size)
#
# train_dist_dataset = strategy.experimental_distribute_datasets_from_function(train_dataset)
# val_dist_dataset = strategy.experimental_distribute_datasets_from_function(val_dataset)

# In[ ]:


# Fit the model
history = model.fit(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                    validation_data=val_data_gen, validation_steps=nb_valid_steps,
                    callbacks=callbacks)#,[chkpt, PrintLR()],
#                     #class_weight={0:1.0, 1:0.4})


print("#--#--"*10,"\n\nTraining Completed \n\n")

# In[ ]:


h = history
fig = plt.figure()
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.plot(np.argmax(h.history["accuracy"]),
         np.max(h.history["val_accuracy"]),
         marker="x", color="b", label="best model")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train-accuracy', 'val-accuracy'], loc='upper left')
#plt.savefig("/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_acc.png", format='png')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_acc-'+FileTime+'.png', format='png')
plt.close(fig)
#
# # In[ ]:
#
#
h = history
fig = plt.figure()
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.plot(np.argmin(h.history["loss"]),
         np.min(h.history["val_loss"]),
         marker="x", color="r", label="best model")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train-loss', 'val-loss'], loc='upper left')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_loss-'+FileTime+'.png', format='png')
plt.close()

# In[ ]:


# modelPath = './xray-best-model/best_model/best_model_'+datetime.datetime.now().strftime("%m-%d-%Y-%I-%M")+'.h5'
weightPath = '/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/xray-best-model/best_model/best_model_para-tune_'+FileTime+'.hdf5'
# model.save(modelPath)
model.save_weights(weightPath)


# In[ ]:


# print("modelPath: ", modelPath)
print("#--#--"*10,"\n\nweightPath: ", weightPath)


# In[ ]:


model.load_weights(weightPath)


# In[ ]:


# Preparing test data
normal_cases_dir = test_dir / 'NORMAL'
pneumonia_cases_dir = test_dir / 'PNEUMONIA'

normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

test_data = []
test_labels = []

for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(0, num_classes=2)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    label = to_categorical(1, num_classes=2)
    test_data.append(img)
    test_labels.append(label)


test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("#--#--"*10,"\n\nTotal number of test examples: ", test_data.shape)
print("#--#--"*10,"\n\nTotal number of labels:", test_labels.shape)


# In[ ]:


# Evaluation on test dataset
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)
print("#--#--"*10,"\n\nLoss on test set: ", test_loss)
print("#--#--"*10,"\n\nAccuracy on test set: ", test_score)


# In[ ]:


# Get predictions
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(test_labels, axis=-1)

print("#--#--"*10,"\n\n",orig_test_labels.shape)
print("#--#--"*10,"\n\n",preds.shape)


# In[ ]:


# Get the confusion matrix
cm  = confusion_matrix(orig_test_labels, preds)
fig = plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
#plt.savefig("/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_confusion-mat.png", format='png')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_confusion-mat-'+FileTime+'.png', format='png')
plt.close()

# In[ ]:


# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("#--#--"*10,"\n\nRecall of the model is {:.2f}".format(recall))
print("#--#--"*10,"\n\nPrecision of the model is {:.2f}".format(precision))


# In[ ]:

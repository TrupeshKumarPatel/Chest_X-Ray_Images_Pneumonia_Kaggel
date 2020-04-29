#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_cell_magic('javascript', '', 'IPython.notebook.clear_all_output();')


# In[2]:

#
# get_ipython().run_line_magic('reset', '-f')
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
#
# get_ipython().run_line_magic('who', '')
#

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
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
os.environ['AUTOGRAPH_VERBOSITY'] = "10"
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(0)

import sys
import datetime
import time

print("Python version: ", sys.version)
print("Version info.: ", sys.version_info)
print("TensorFlow version: ", tf.__version__)
print("TensorFlow.Keras version : ", tf.keras.__version__)


# In[4]:



# Turn interactive plotting off
# plt.ioff()
# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(111)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
config.gpu_options.visible_device_list='0,1,2,3'
# config.gpu_options.visible_device_list='0,1'

# Set the random seed in tensorflow at graph level
tf.compat.v1.set_random_seed(111)

# Set the session in tensorflow
sess = tf.compat.v1.Session(config=config)

# Set the session in keras
tf.compat.v1.keras.backend.set_session(sess)

# tf.debugging.set_log_device_placement(True)
strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

print("#--#--"*10)
print('Number of devices: {}\n'.format(strategy.num_replicas_in_sync))

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# Make the augmentation sequence deterministic
aug.seed(111)

AUTOTUNE = tf.data.experimental.AUTOTUNE


# In[5]:


FileTime = str(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
print("#--#--"*10, "\nFile time: ", FileTime, "\n\n")

# Define path to the data directory
data_dir = Path('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Data/new_chest_xray/')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'

# Path to validation directory
val_dir = data_dir / 'val'

# Path to test directory
test_dir = data_dir / 'test'

# Define path to the data directory
val_2_data_dir = Path('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Data/chest_xray/chest_xray/')

# Path to test directory
val_dir_2 = val_2_data_dir / 'test'

# In[6]:



train_list_ds = tf.data.Dataset.list_files(str(train_dir/"*"/"*"))
test_list_ds = tf.data.Dataset.list_files(str(test_dir/"*"/"*"))
# val_list_ds = tf.data.Dataset.list_files(str(val_dir/"*"/"*"))
val_list_ds = tf.data.Dataset.list_files(str(val_dir_2/"*"/"*"))

# for f in train_list_ds.take(1):
#     print(f.numpy())
# In[7]:


CLASS_NAMES = np.array(['NORMAL','PNEUMONIA'])
def get_label(file_path):
    label = None
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    print(parts)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES

def decode_img(img, IMG_WIDTH=299, IMG_HEIGHT=299):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# In[8]:


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')


# In[9]:



train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_labeled_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# for image, label in labeled_ds.take(1):
#     print("Image shape: ", image.numpy().shape)
#     print("Label: ", label.numpy())


# In[10]:


# BATCH_SIZE = 32
# BUFFER_SIZE = 1000
IMG_SIZE = 299
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

nb_epochs = 20
BUFFER_SIZE = 5

BATCH_SIZE_PER_REPLICA = 32
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# In[11]:


# with strategy.scope():
train_batches = train_labeled_ds.repeat().batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
validation_batches = val_labeled_ds.repeat().batch(BATCH_SIZE)
test_batches = test_labeled_ds.batch(BATCH_SIZE)


# In[12]:


# train_size = get_ipython().getoutput('ls $train_dir/*/*.jpeg | wc -l')
# test_size = get_ipython().getoutput('ls $test_dir/*/*.jpeg | wc -l')
# val_size = get_ipython().getoutput('ls $val_dir/*/*.jpeg | wc -l')
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'
train_size = len(list(normal_cases_dir.glob('*.jpeg'))) + len(list(pneumonia_cases_dir.glob('*.jpeg')))
normal_cases_dir = test_dir / 'NORMAL'
pneumonia_cases_dir = test_dir / 'PNEUMONIA'
test_size = len(list(normal_cases_dir.glob('*.jpeg'))) + len(list(pneumonia_cases_dir.glob('*.jpeg')))
normal_cases_dir = val_dir / 'NORMAL'
pneumonia_cases_dir = val_dir / 'PNEUMONIA'
val_size = len(list(normal_cases_dir.glob('*.jpeg'))) + len(list(pneumonia_cases_dir.glob('*.jpeg')))

# train_size, test_size, val_size = int(train_size[0]), int(test_size[0]), int(val_size[0])

print("#--#--"*10,"\ntraining: {},\nvalidation: {},\ntest: {}".format(train_size,
                                                                        val_size,
                                                                        test_size))

# train_validation_concat = train_batches.concatenate(validation_batches)
# concat_step = (train_size + val_size)//BATCH_SIZE
# print(concat_step)
# l = []
# for _, y in train_validation_concat.take(concat_step+2):
#     for i in y:
#         l.append(i)
#
# y_concat = tf.convert_to_tensor(np.asarray(l, dtype=np.bool),  dtype=tf.bool)
# print(y_concat.shape)for image_batch, label_batch in train_batches.take(1):
#     pass
#
# print(image_batch.shape)
# In[13]:


def print_inventory(inventory_name, dct):
    print('%s :' %(inventory_name))
    for item, amount in dct.items():  # dct.iteritems() in Python 2
        print('%15s : %s' % (item, amount))

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}\n'.format(epoch + 1, model.optimizer.lr.numpy()))


# In[14]:
# def scheduler(epoch):
#     if epoch < 10:
#         return 0.001
#     else:
#         return 0.001 * tf.math.exp(0.1 * (10 - epoch)

learning_rate = 1e-4
callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max', min_delta=0.0001),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max'),
    EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),

    # EarlyStopping(  monitor='val_loss', min_delta=1e-3, patience=5, verbose=1,
    #                 mode='auto', baseline=None, restore_best_weights=True),
    ModelCheckpoint(filepath='/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/best_model_todate_python/'+FileTime+'/',
                    monitor='val_auc', verbose=1, mode='max',
                    save_best_only=True, save_weights_only=True),
    PrintLR()
]


# In[15]:


# Define the number of training steps
nb_train_steps = train_size//BATCH_SIZE

# Define the number of validation steps
nb_valid_steps = val_size//BATCH_SIZE

# Define the number of validation steps
nb_test_steps = test_size//BATCH_SIZE

print("#--#--"*10,"\n\nNumber of training and validation steps: {} and {}".format(nb_train_steps,
                                                                                  nb_valid_steps))

print("#--#--"*10,  "\n BATCH_SIZE_PER_REPLICA = ", BATCH_SIZE_PER_REPLICA,
                    "\n BATCH_SIZE = ", BATCH_SIZE,
                    "\n EPOCHS = ", nb_epochs)


# In[16]:


# Create the base model from the pre-trained model MobileNet V2
with strategy.scope():
    base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

# with strategy.scope():
#     feature_batch = base_model(image_batch)
# #     feature_batch = base_model.predict(train_validation_concat, verbose=1)
#
# print(feature_batch.shape)

# In[17]:
# base_model.summary()

print("#--#--"*20)
print("Model Name:", base_model.name)
print("#--#--"*20)

reg_l2 = 0.0001
with strategy.scope():
    inputs = base_model.output
    flat = Flatten(name='flatten')(inputs)
    dense_1 = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='fc1')(flat)
    drop_1 = Dropout(0.7, name='dropout1')(dense_1)
    dense_2 = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='fc2')(drop_1)
    drop_2 = Dropout(0.5, name='dropout2')(dense_2)
    predict = Dense(2, activation='softmax', name='fc3')(drop_2)

    model = Model(inputs=base_model.input, outputs=predict)
    model._name = "FrozenModel"
    opt = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True, clipnorm=1., decay=1e-5)
    # opt = tf.keras.optimizers.Nadam()
#     model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC()])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model.summary()
# In[20]:


print("#--#--"*10)
print_inventory("Optimizer", opt.get_config())
# print("#--#--"*10)
# print_inventory("Early Stopping", opt.get_config())
# print("#--#--"*10)
# print_inventory("Reduce LR On Plateau", opt.get_config())

print('\nReduce LR On Plateau :\n%15s : %s,\n%15s : %s,\n%15s : %s,\n%15s : %s'%('Monitor', callbacks[0].monitor,
                                                                                 'Factor', callbacks[0].factor,
                                                                                 'Mode', callbacks[0].mode,
                                                                                 'Patience', callbacks[0].patience) )

# print('\n Early Stopping:\n%15s : %s,\n%15s : %s,\n%15s : %s' %('Monitor', callbacks[1].monitor,
#                                                                 'MinDelta', callbacks[1].min_delta,
#                                                                 'patience', callbacks[1].patience) )


# In[ ]:


with strategy.scope():
    history = model.fit(train_batches, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                        validation_data=validation_batches, validation_steps=nb_valid_steps,
                        callbacks=callbacks, verbose=1)#,[chkpt, PrintLR()],
                        # class_weight={0:1.0, 1:0.4})


# In[ ]:
# modelPath = '/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/xray-best-model/best_model/best_model_'+FileTime+'.h5'
weightPath = '/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/xray-best-model/best_model/best_model_para-tune_'+FileTime+'.hdf5'
# model.save(modelPath)
model.save_weights(weightPath)
# print("modelPath: ", modelPath)
print("#--#--"*10,"\n\nweightPath: ", weightPath)

# print("#--#--"*10,"\n\nTraining Completed \n\n")
# print("history items : ", history.history.keys())


with strategy.scope():
    base_model_reLoad = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                          include_top=False,
                                                          weights='imagenet')
    for layer in base_model_reLoad.layers:
        base_model_reLoad.trainable = True

# base_model_reLoad.summary()

with strategy.scope():
    inputs = base_model_reLoad.output

    x = Flatten(name='flatten')(inputs)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_l2), name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(2, activation='softmax', name='fc3')(x)

    model_reLoad = Model(inputs=base_model_reLoad.input, outputs=x)
    model_reLoad._name = "UnfrozenModel"
    opt = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True, clipnorm=1., decay=1e-5)
    # opt = tf.keras.optimizers.Nadam()
#     model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy', tf.keras.metrics.AUC()])
    model_reLoad.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  optimizer=opt, metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

model_reLoad.summary()

print("#--#--"*10,"\n\nLoading weightPath: ", weightPath)
with strategy.scope():
    model_reLoad.load_weights(weightPath)

callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, verbose=1, mode='max', min_delta=0.0001),
    # ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max'),
    # EarlyStopping(monitor='val_auc', min_delta=1e-3, patience=5, mode='max', restore_best_weights=True),

    # EarlyStopping(  monitor='val_loss', min_delta=1e-3, patience=5, verbose=1,
    #                 mode='auto', baseline=None, restore_best_weights=True),
    ModelCheckpoint(filepath='/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/best_model_todate_python/'+FileTime+'/',
                    monitor='val_auc', verbose=1, mode='max',
                    save_best_only=True, save_weights_only=True),
    PrintLR()
]

with strategy.scope():
    history_reLoad = model_reLoad.fit(train_batches, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                                      validation_data=validation_batches, validation_steps=nb_valid_steps,
                                      callbacks=callbacks, verbose=1)#,[chkpt, PrintLR()],
                                    # class_weight={0:1.0, 1:0.4})


# In[ ]:

print("#--#--"*10,"\n\nTraining Completed \n\n")
print("history items : ", history_reLoad.history.keys())

h = history_reLoad
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
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/plots/FinalRun_2/acc/python_acc-'+FileTime+'_reLoad.png', format='png')
plt.close(fig)


# In[ ]:


h = history_reLoad
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
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/plots/FinalRun_2/loss/python_loss-'+FileTime+'_reLoad.png', format='png')
plt.close()


# In[ ]:


h = history_reLoad
fig = plt.figure()
plt.plot(h.history['auc'])
plt.plot(h.history['val_auc'])
plt.plot(np.argmax(h.history["auc"]),
         np.max(h.history["val_auc"]),
         marker="x", color="b", label="best model")
plt.title('model AUC (Area under the curve)')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train-AUC', 'val-AUC'], loc='upper left')
#plt.savefig("/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_acc.png", format='png')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/plots/FinalRun_2/auc/python_auc-'+FileTime+'_reLoad.png', format='png')
plt.close(fig)


# In[ ]:


# modelPath = '/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/xray-best-model/best_model/best_model_'+FileTime+'.h5'
weightPath_reLoad = '/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/xray-best-model/best_model/best_model_para-tune_'+FileTime+'_reLoad.hdf5'
# model.save(modelPath)
model_reLoad.save_weights(weightPath_reLoad)
# print("modelPath: ", modelPath)
print("#--#--"*10,"\n\nweightPath_reLoad: ", weightPath_reLoad)

# In[ ]:


with strategy.scope():
    model_reLoad.load_weights(weightPath_reLoad)


# In[ ]:

##################################################################################################################################################
#           TEST - 1
##################################################################################################################################################
# Evaluation on test dataset
with strategy.scope():
    test_loss, test_score, test_auc = model_reLoad.evaluate(test_batches)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)
print("AUC on test set: ", test_auc)

pred_test_labels=[]
orig_test_labels=[]
for X_sample, Y_sample in test_batches.take(4):
    print(X_sample.shape, Y_sample.shape)
#     with strategy.scope():
    pred = model_reLoad.predict(X_sample)
    for pY, oY in zip(pred, Y_sample):
        pred_test_labels.append(pY)
        orig_test_labels.append(oY)

pred_test_labels = np.argmax(pred_test_labels, axis=-1)
orig_test_labels = np.argmax(orig_test_labels, axis=-1)

print("#--#--"*10,"\n\n",orig_test_labels.shape)
print(pred_test_labels.shape)

# Get the confusion matrix
cm  = confusion_matrix(orig_test_labels, pred_test_labels)
fig = plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
#plt.savefig("/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_confusion-mat.png", format='png')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/plots/FinalRun_2/confusion-mat/python_confusion-mat-'+FileTime+'_reLoad_TEST-1.png', format='png')
plt.close()

# In[ ]:


# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall/(precision+recall))
print("#--#--"*10,"\n\nRecall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("F1-score: {}".format(f1_score))

##################################################################################################################################################
#           TEST - 2
##################################################################################################################################################

# Define path to the data directory
test_2_data_dir = Path('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Data/chest_xray/chest_xray/')

# Path to test directory
test_dir_2 = test_2_data_dir / 'test'

with strategy.scope():
    test_2_list_ds = tf.data.Dataset.list_files(str(test_dir_2/"*"/"*"))

with strategy.scope():
    test_2_labeled_ds = test_2_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

test_2_batches = test_2_labeled_ds.batch(BATCH_SIZE)

# Evaluation on test dataset
with strategy.scope():
    test_loss, test_score, test_auc = model_reLoad.evaluate(test_2_batches)
print("Loss on test 2 set: ", test_loss)
print("Accuracy on test 2 set: ", test_score)
print("AUC on test 2 set: ", test_auc)


pred_test_labels=[]
orig_test_labels=[]
for X_sample, Y_sample in test_2_batches.take(4):
    print(X_sample.shape, Y_sample.shape)
#     with strategy.scope():
    pred = model_reLoad.predict(X_sample)
    for pY, oY in zip(pred, Y_sample):
        pred_test_labels.append(pY)
        orig_test_labels.append(oY)

pred_test_labels = np.argmax(pred_test_labels, axis=-1)
orig_test_labels = np.argmax(orig_test_labels, axis=-1)

print("#--#--"*10,"\n\n",orig_test_labels.shape)
print(pred_test_labels.shape)


# Get the confusion matrix
cm  = confusion_matrix(orig_test_labels, pred_test_labels)
fig = plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
#plt.savefig("/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/python_confusion-mat.png", format='png')
plt.savefig('/data/user/tr27p/Courses/CS765-DeepLearning/FinalProject/Chest_X-Ray_Images_Pneumonia/Python/plots/FinalRun_2/confusion-mat/python_confusion-mat-'+FileTime+'_reLoad_TEST-2.png', format='png')
# plt.show()
plt.close()

# In[ ]:


# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("#--#--"*10,"\n\nRecall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
print("F1-score: {}".format(2*precision*recall/(precision+recall)))

print("#--#--"*20)
print("FILE TIME:", FileTime)
print("Model Name:", base_model.name)
print("#--#--"*20)

#!/usr/bin/env python
# coding: utf-8

# #### Convolutional Neural Network

# In[1]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from imutils import paths
import matplotlib.pyplot as plt
import argparse
import os
os.chdir('D:\Datasets\Fire uk')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# rescale and resize - very imp preprocessing step


train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    r'Dataset\Training',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
   r'Dataset\Training', # same directory as training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # set as validation data



# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator  = test_datagen.flow_from_directory('Dataset\Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[9]:


bModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224,3 )))  #base_Model
hModel = bModel.output #head_Model
hModel = AveragePooling2D(pool_size=(4, 4))(hModel)
hModel = Flatten(name="flatten")(hModel)
hModel = Dense(64, activation="relu")(hModel)
hModel = Dropout(0.5)(hModel)
hModel = Dense(2, activation="softmax")(hModel)
model = Model(inputs=bModel.input, outputs=hModel)
for layer in bModel.layers:
    layer.trainable = False


# In[10]:


from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint = ModelCheckpoint(r"models\model2.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]


# In[11]:


# Part 3 - Training the CNN

# Compiling the CNN
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
history=model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // 32,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 32,
    callbacks=callbacks,
    epochs = 10)

loss_tr, acc_tr = model.evaluate_generator(train_generator)

print(loss_tr)
print(acc_tr)


loss_vr, acc_vr = model.evaluate_generator(validation_generator)

print(loss_vr)
print(acc_vr)


loss_test, acc_test= model.evaluate_generator(test_generator)


print(loss_test)
print(acc_test)

import pandas as pd
loss=[loss_tr,loss_vr,loss_test]
accuracy=[acc_tr * 100,acc_vr * 100,acc_test * 100]
df=pd.DataFrame(loss,columns=['loss'],index=['Training','Validation','Test'])
df['Accuracy']=accuracy
import dataframe_image as dfi
dfi.export(df, r'Outputs\results.png')


loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(0,8)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Outputs\Training_vs_Validation_accuracy.png')
plt.show()


loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,8)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Outputs\Training_vs_Validation_loss.png')
plt.show()


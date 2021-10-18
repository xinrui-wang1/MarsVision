#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import math
import IPython.display as display
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.imagenet_utils import decode_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ## EDA

# In[2]:


def readDatasetDF(datasetType, rootDir='data/'):
    """
    dataset Type: "train", "val", "test" 
    Return:
        dataframe with image name/label
    """
    df = pd.read_csv('{}/{}-calibrated-shuffled.txt'.format(rootDir, datasetType), header=None, delimiter = " ")
    df[0] = rootDir+"/"+df[0]
    return df

df_tr = readDatasetDF('train')
df_val = readDatasetDF('val')
df_test = readDatasetDF('test')
df_tr.head()


# In[3]:


print(df_tr.shape)
print(df_val.shape)
print(df_test.shape)


# In[4]:


df_tr[1].value_counts()


# In[5]:


class_names = pd.read_csv('data/msl_synset_words-indexed.txt', delimiter="      ", header=None, engine='python').set_index(0).to_dict()[1]
for i in class_names.keys():
    class_names[i] = class_names[i].strip()
class_names


# In[6]:


df_tr[2] = df_tr[1].map(class_names)
df_val[2] = df_val[1].map(class_names)
df_test[2] = df_test[1].map(class_names)


# In[7]:


img_paths = df_tr.iloc[:, 0].tolist()
rootDir = 'msl-images/'
img = cv2.imread(img_paths[0])


# In[8]:


img_paths = df_tr.loc[df_tr[1] == 8][0].iloc[0]
rootDir = 'msl-images/'
img = cv2.imread(img_paths)
plt.imshow(img)


# ## Preprocessing

# In[9]:


#Combine all datasets
frames = [df_tr, df_val, df_test]
result = pd.concat(frames)
result.reset_index(drop = True, inplace = True)


# In[10]:


#Split the data again
X = result[0]
Y = result[1]

X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify = Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=0.25, random_state=1, stratify = Y_train_temp)


# In[11]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# In[12]:


train_df = pd.concat([X_train, Y_train], axis=1)
val_df = pd.concat([X_val, Y_val], axis=1)
test_df = pd.concat([X_test, Y_test], axis=1)


# In[13]:


#Check all classes are represented
print(sorted(train_df[1].unique()))
print(sorted(val_df[1].unique()))
print(sorted(test_df[1].unique()))


# In[14]:


train_df[1] = train_df[1].replace(23,22).replace(24,23)
val_df[1] = val_df[1].replace(23,22).replace(24,23)
test_df[1] = test_df[1].replace(23,22).replace(24,23)


# In[15]:


#Check all classes are represented
print(sorted(train_df[1].unique()))
print(sorted(val_df[1].unique()))
print(sorted(test_df[1].unique()))


# In[ ]:


class_names[22] = 'turret'
class_names[23] = 'wheel'
class_names.pop(24)


# In[16]:


train_df.to_csv('data/train_df_new.txt', header=None, index=None, sep=' ', mode='w')
val_df.to_csv('data/val_df_new.txt', header=None, index=None, sep=' ', mode='w')
test_df.to_csv('data/test_df_new.txt', header=None, index=None, sep=' ', mode='w')


# In[17]:


#Prepare the dataset
batch_size = 32
img_height = 256
img_width = 256
path = 'data/'

train_dir = path + '/train_df_new.txt'
validation_dir = path + '/val_df_new.txt'
test_dir = path + '/test_df_new.txt'

def read_dataset(dir):
    df = pd.read_csv(dir, header=None, delimiter = " ")
    df[0] = df[0]
    image_paths = df[0].values
    labels = df[1].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    return dataset

train_ds = read_dataset(train_dir)
val_ds = read_dataset(validation_dir)
test_ds = read_dataset(test_dir)

def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    #image = (image / 255.0)
    return image, label 

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(read_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(read_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(read_image, num_parallel_calls=AUTOTUNE)


# In[18]:


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds =  configure_for_performance(test_ds)


# ## Transfer Learning

# In[24]:


# Baseline Resnet50 model
base_model = tf.keras.applications.ResNet50(
    include_top=False, input_shape=(256, 256, 3), weights='imagenet'
)


# In[25]:


# Freeze the base
base_model.trainable = False


# In[26]:


base_model.summary()


# In[27]:


num_classes = 24

inputs = tf.keras.layers.Input([256, 256, 3])
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(25, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[28]:


initial_epochs = 5

history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)


# In[29]:


# Learning CUrve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[30]:


model.evaluate(test_ds)


# In[31]:


def get_confusion_matrix(model, ds):
    all_predictions = np.array([])
    all_labels = np.array([])
    for x, y in ds:
        predictions = model.predict(x)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        all_labels = np.concatenate([all_labels, y])
    return tf.math.confusion_matrix(all_predictions, all_labels)


# In[32]:


cf_matrix = get_confusion_matrix(model, test_ds)


# In[33]:


plt.figure(figsize=(13,13))
plt.imshow(cf_matrix.numpy(), cmap=plt.cm.Blues)

for i in range(24):
    for j in range(24):
        c = cf_matrix.numpy()[j,i]
        plt.text(i, j, str(c), va='center', ha='center')

ax = plt.xticks(range(24), np.sort(test_df[1].unique()))
ax = plt.yticks(range(24), np.sort(test_df[1].unique()))


# In[ ]:





# In[ ]:





# ## Model with image augmentation

# In[34]:


RESIZE = 255
BATCH_SIZE = 32

data_augmentation = tf.keras.Sequential([        
    tf.keras.layers.experimental.preprocessing.Rescaling(1./RESIZE),                          
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])


# In[35]:


epochs = 100
input_shape = (256,256,3)
# Create base model
base_model = tf.keras.applications.ResNet50V2(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    pooling=None,
    classes=24,
    classifier_activation="softmax",
)
# Freeze base model
#base_model.trainable = False
base_model.trainable = True

fine_tune_at = 130

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# In[36]:


# Create new model on top.
inputs = tf.keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(25)(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


# In[37]:


base_model.summary()


# In[38]:


print("Number of layers in the base model: ", len(base_model.layers))


# In[39]:


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, min_lr=1e-6, mode='min')


# In[40]:


# fit_generator
history = model.fit(
    train_ds, 
    epochs= 5, 
    batch_size=BATCH_SIZE, 
    steps_per_epoch=None, 
    validation_data=val_ds, 
    validation_steps=None,
    callbacks=lr_reducer
) 


# In[41]:


# Learning CUrve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[44]:


model.evaluate(test_ds)


# In[45]:


def get_confusion_matrix(model, ds):
    all_predictions = np.array([])
    all_labels = np.array([])
    for x, y in ds:
        predictions = model.predict(x)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        all_labels = np.concatenate([all_labels, y])
    return tf.math.confusion_matrix(all_predictions, all_labels)


# In[46]:


cf_matrix = get_confusion_matrix(model, test_ds)


# In[47]:


plt.figure(figsize=(13,13))
plt.imshow(cf_matrix.numpy(), cmap=plt.cm.Blues)

for i in range(24):
    for j in range(24):
        c = cf_matrix.numpy()[j,i]
        plt.text(i, j, str(c), va='center', ha='center')

ax = plt.xticks(range(24), np.sort(test_df[1].unique()))
ax = plt.yticks(range(24), np.sort(test_df[1].unique()))


# ## Eliminating under-represented classes

# The data has few under-sampled classes (under 80), we would like to remove them
# 

# In[51]:


result[1].value_counts(dropna=False)


# In[52]:


#omitted classes with less than 80 samples
omit = [0, 20, 19, 13, 6, 4, 2, 11, 1, 18]
omit_index = []
for index, row in result.iterrows():
    if row[1] in omit:
        omit_index.append(index)
        
result.drop([x for x in omit_index], inplace = True, axis = 0)
result.reset_index(drop = True, inplace = True)


# In[53]:


result.head(2)


# In[54]:


#Map labels to new numbers
label_map = dict(zip(result[1].unique(), list(range(14))))
new_label_l = []
for x in result.itertuples():
    new_label_l.append(label_map[x[2]])
result[1] = new_label_l


# In[55]:


#Split the data again
X = result[0]
Y = result[1]
X_train_temp, X_test, Y_train_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify = Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_temp, Y_train_temp, test_size=0.25, random_state=1, stratify = Y_train_temp)


# In[56]:


print(X_train.shape)
print(X_val.shape)
print(X_test.shape)


# In[57]:


train_df = pd.concat([X_train, Y_train], axis=1)
val_df = pd.concat([X_val, Y_val], axis=1)
test_df = pd.concat([X_test, Y_test], axis=1)


# In[58]:


#Check all classes are represented
print(sorted(train_df[1].unique()))
print(sorted(val_df[1].unique()))
print(sorted(test_df[1].unique()))


# In[69]:


train_df.to_csv('data/train_df_no_under80.txt', header=None, index=None, sep=' ', mode='w')
val_df.to_csv('data/val_df_no_under80.txt', header=None, index=None, sep=' ', mode='w')
test_df.to_csv('data/test_df_no_under80.txt', header=None, index=None, sep=' ', mode='w')


# In[70]:


batch_size = 32
img_height = 256
img_width = 256
path = 'data'

train_dir = path + '/train_df_no_under80.txt'
validation_dir = path + '/val_df_no_under80.txt'
test_dir = path + '/test_df_no_under80.txt'

def read_dataset(dir):
    df = pd.read_csv(dir, header=None, delimiter = " ")
    df[0] = df[0]
    image_paths = df[0].values
    labels = df[1].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    return dataset

train_ds = read_dataset(train_dir)
val_ds = read_dataset(validation_dir)
test_ds = read_dataset(test_dir)

def read_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    #image = (image / 255.0)
    return image, label 

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(read_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(read_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(read_image, num_parallel_calls=AUTOTUNE)


# In[71]:


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds =  configure_for_performance(test_ds)


# ### Rerun the model with image augmetation

# In[72]:


RESIZE = 255
BATCH_SIZE = 32


# In[73]:


data_augmentation = tf.keras.Sequential([        
    tf.keras.layers.experimental.preprocessing.Rescaling(1./RESIZE),                          
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])


# In[74]:


epochs = 100
input_shape = (256,256,3)
# Create base model
base_model = tf.keras.applications.ResNet50V2(
    input_shape=input_shape,
    include_top=False,
    weights="imagenet",
    pooling=None,
    classes=25,
    classifier_activation="softmax",
)
# Freeze base model
#base_model.trainable = False
base_model.trainable = True

fine_tune_at = 130

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# In[80]:


# Create new model on top.
inputs = tf.keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(14)(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)


# In[81]:


base_model.summary()


# In[82]:


print("Number of layers in the base model: ", len(base_model.layers))


# In[83]:


lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, min_lr=1e-6, mode='min')


# In[84]:


# fit_generator
history = model.fit(
    train_ds, 
    epochs= 5, 
    batch_size=BATCH_SIZE, 
    steps_per_epoch=None, 
    validation_data=val_ds, 
    validation_steps=None,
    callbacks=lr_reducer
) 


# In[85]:


# Learning CUrve
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[86]:


model.evaluate(test_ds)


# In[87]:


def get_confusion_matrix(model, ds):
    all_predictions = np.array([])
    all_labels = np.array([])
    for x, y in ds:
        predictions = model.predict(x)
        predictions = np.argmax(predictions, axis = 1)
        all_predictions = np.concatenate([all_predictions, predictions])
        all_labels = np.concatenate([all_labels, y])
    return tf.math.confusion_matrix(all_predictions, all_labels)


# In[88]:


cf_matrix = get_confusion_matrix(model, test_ds)


# In[89]:


plt.figure(figsize=(13,13))
plt.imshow(cf_matrix.numpy(), cmap=plt.cm.Blues)

for i in range(14):
    for j in range(14):
        c = cf_matrix.numpy()[j,i]
        plt.text(i, j, str(c), va='center', ha='center')

ax = plt.xticks(range(14), np.sort(test_df[1].unique()))
ax = plt.yticks(range(14), np.sort(test_df[1].unique()))


# In[ ]:





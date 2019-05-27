# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:10:35 2019

@author: abhis
"""

from __future__ import division, print_function
from PIL import Image
import glob
import numpy as np
import cv2
from keras import backend as K
from keras.applications import vgg16
from keras.layers import Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential, Model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from random import shuffle
from scipy.misc import imresize
import itertools
import matplotlib.pyplot as plt
import os
#%matplotlib inline


def process():
    image_list = []
    for filename in glob.glob('C:/Users/abhis/Desktop/us/Courses/AML 674/P2/AND_images/*.png'): #assuming gif
        im=cv2.imread(filename)
        image_list.append(im)
    #print(image_list)
    image_list=np.array(image_list)
    return image_list

image_list=process()
np.random.shuffle(image_list)
tr,val,test=image_list[:9700],image_list[9700:11774],image_list[11774:]
#print(test)
#print(len(test))

for i in range(len(tr)-1):
    left_image = tr[i]
    right_image = tr[i+1]


DATA_DIR = ("C:/Users/abhis/Desktop/us/Courses/AML 674/P2/")
IMAGE_DIR = os.path.join(DATA_DIR, "AND_images/")
#print(IMAGE_DIR)

def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return gname + pname + ".png"
    
def create_triples(image_dir):
    img_groups = {}
    for img_file in os.listdir(image_dir):
        prefix, suffix = img_file.split(".")
        gid, pid = prefix[0:4], prefix[4:]
        if gid in img_groups:
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    pos_triples, neg_triples = [], []
    # positive pairs are any combination of images in same group
    for key in img_groups.keys():
        triples = [(key + x[0] + ".png", key + x[1] + ".png", 1) 
                 for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # need equal number of negative examples
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    #pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    shuffle(neg_triples)
    x = int(len(pos_triples)*0.8)
    y = int(len(neg_triples)*0.8)
    x1 = int(len(pos_triples)*0.1)
    y1 = int(len(neg_triples)*0.1)
    #print(neg_triples[:5])
    #print(pos_triples[:5])
    a = pos_triples[:x]
    b = neg_triples[:y]
    pos_triples_split_train = a+b
    #print(pos_triples_split_train)
    pos_triples_split_val = pos_triples[x:x+x1]+(neg_triples[y:y+y1])
    pos_triples_split_test = pos_triples[x+x1:]+(neg_triples[y+y1:])
    
    #print(pos_triples_split_train[0:5])
    return pos_triples_split_train, pos_triples_split_val, pos_triples_split_test

pos_triples_split_train, pos_triples_split_val, pos_triples_split_test = create_triples(IMAGE_DIR)
#print("# image triples:", len(triples_data))
#[x for x in triples_data[0:5]]


def load_image(image_name):
    if image_name not in image_cache:
        image = plt.imread(os.path.join(IMAGE_DIR, image_name)).astype(np.float32)
        image = imresize(image, (64, 64))
        image = np.divide(image, 255)
        image_cache[image_name] = image
    return image_cache[image_name]
    
def generate_image_triples_batch(image_triples, batch_size, shuffle=False):
    while True:
        # loop once per epoch
        if shuffle:
            indices = np.random.permutation(np.arange(len(image_triples)))
        else:
            indices = np.arange(len(image_triples))
        shuffled_triples = [image_triples[ix] for ix in indices]
        num_batches = len(shuffled_triples) // batch_size
        for bid in range(num_batches):
            # loop once per batch
            images_left, images_right, labels = [], [], []
            batch = shuffled_triples[bid * batch_size : (bid + 1) * batch_size]
            for i in range(batch_size):
                lhs, rhs, label = batch[i]
                images_left.append(load_image(lhs))
                images_right.append(load_image(rhs))              
                labels.append(label)
            Xlhs = np.array(images_left)
            Xrhs = np.array(images_right)
            Y = np_utils.to_categorical(np.array(labels), num_classes=2)
            yield ([Xlhs, Xrhs], Y)

            
BATCH_SIZE = 64

#split_point = int(len(triples_data) * 0.7)
#triples_train, triples_test = triples_data[0:split_point], triples_data[split_point:]


def create_base_network(input_shape):
    seq = Sequential()
    # CONV => RELU => POOL
    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten => RELU
    seq.add(Flatten())
    seq.add(Dense(500))
    
    return seq

def cosine_distance(vecs, normalize=False):
    x, y = vecs
    if normalize:
        x = K.l2_normalize(x, axis=0)
        y = K.l2_normalize(x, axis=0)
    return K.prod(K.stack([x, y], axis=1), axis=1)

def cosine_distance_output_shape(shapes):
    return shapes[0]

def compute_accuracy(preds, labels):
    return labels[preds.ravel() < 0.5].mean()


input_shape = (64, 64, 3)
base_network = create_base_network(input_shape)

image_left = Input(shape=input_shape)
image_right = Input(shape=input_shape)

vector_left = base_network(image_left)
vector_right = base_network(image_right)

distance = Lambda(cosine_distance, 
                  output_shape=cosine_distance_output_shape)([vector_left, vector_right])

# fc1 = Dense(512, kernel_initializer="glorot_uniform")(distance)
# fc1 = Dropout(0.2)(fc1)
# fc1 = Activation("relu")(fc1)

fc1 = Dense(128, kernel_initializer="glorot_uniform")(distance)
fc1 = Dropout(0.2)(fc1)
fc1 = Activation("relu")(fc1)

pred = Dense(2, kernel_initializer="glorot_uniform")(fc1)
pred = Activation("softmax")(pred)

model = Model(inputs=[image_left, image_right], outputs=pred)
# model.summary()


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

NUM_EPOCHS = 10 
image_cache = {}
train_gen = generate_image_triples_batch(pos_triples_split_test, BATCH_SIZE, shuffle=True)
val_gen = generate_image_triples_batch(pos_triples_split_val, BATCH_SIZE, shuffle=False)

num_train_steps = len(pos_triples_split_test) // BATCH_SIZE
num_val_steps = len(pos_triples_split_val) // BATCH_SIZE

history = model.fit_generator(train_gen,
                              steps_per_epoch=num_train_steps,
                              epochs=NUM_EPOCHS,
                              validation_data=val_gen,
                              validation_steps=num_val_steps)

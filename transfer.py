import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import bcolz

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras import applications

import cv2
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time
import argparse

DATA_DIR = '/home/chicm/data/planet'
RESULT_DIR = '/home/chicm/data/planet/results'
TRAIN_FEAT = RESULT_DIR + '/trn_feats.dat'
TRAIN_LABEL = RESULT_DIR + '/trn_labels.dat'
TEST_FEAT = RESULT_DIR + '/test_feats.dat'

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
def load_array(fname):
    return bcolz.open(fname)[:]
    
def get_keras_vgg_model():
    model = applications.VGG16(include_top=False, weights='imagenet')
    return model

def get_vgg_model():
    model = get_keras_vgg_model()
    #print(model.summary())
    shape = model.layers[-1].output_shape
    print(shape)
    return model

def gen_vgg_features(input_data, filename):
    model = get_vgg_model()
    feats = model.predict(input_data)
    print(feats.shape)
    save_array(filename, feats)

def gen_train_features():
    x_train, y_train = gen_train_data()
    gen_vgg_features(x_train, TRAIN_FEAT)
    save_array(TRAIN_LABEL, y_train)

def gen_test_features():
    x_test = gen_test_data()
    gen_vgg_features(x_test, TEST_FEAT)

def gen_train_data():
    x_train = []   
    y_train = []
    df_train = pd.read_csv(DATA_DIR+'/train.csv')
    
    flatten = lambda l:[item for sublist in l for item in sublist]
    labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    for f, tags in tqdm(df_train.values, miniters=5000):
        fn = DATA_DIR+'/train-jpg/'+f+'.jpg'
        img = cv2.imread(fn) 
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, (256,256)))
        y_train.append(targets)

    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.uint8)
    
    return x_train, y_train

def gen_test_data():
    x_test = []
    df_test = pd.read_csv(DATA_DIR+'/sample_submission.csv')
    for f, tags in tqdm(df_test.values, miniters=5000):
        fn = DATA_DIR+'/test-jpg/'+f+'.jpg'
        img = cv2.imread(fn)
        x_test.append(cv2.resize(img, (256, 256)))
    x_test = np.array(x_test, np.float32)
    return x_test

def get_bn_layers(input_shape):
    return [
        #MaxPooling2D(input_shape = input_shape),
        Flatten(input_shape=input_shape),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(17, activation='softmax')
    ]

def get_bn_model(input_shape=(8,8,512)):
    bn_model = Sequential(get_bn_layers(input_shape))
    bn_model.compile(Adam(lr=0.001), loss = 'binary_crossentropy', metrics=['accuracy'])
    return bn_model

def train():
    split_index = 35000
    feat = load_array(TRAIN_FEAT)
    y_train = load_array(TRAIN_LABEL)
    print(feat.shape)

    trn_feat = feat[:split_index]
    trn_label = y_train[:split_index]

    val_feat = feat[split_index:]
    val_label = y_train[split_index:]

    model = get_bn_model(feat.shape[1:])
    model.fit(trn_feat, trn_label, batch_size=32, validation_data=(val_feat, val_label), epochs = 10)

def get_conv_layers(input_shape):
    return [    
        Conv2D(24, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(24, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(48, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(48, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(17, activation='sigmoid')
    ]

def train_conv_model():
    x_train, y_train = gen_train_data()
    split_index = 35000
    X_train = x_train[:split_index]
    Y_train = y_train[:split_index]
    
    X_val = x_train[split_index:]
    Y_val = y_train[split_index:]

    model = get_conv_model(x_train.shape[1:])
    model.fit(X_train, Y_train, batch_size=32, validation_data=(X_val, Y_val), epochs = 20)

def get_conv_model(input_shape):
    model = Sequential(get_conv_layers(input_shape))
    model.compile(Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--gentrain", action='store_true',help="generate train features")
parser.add_argument("--gentest", action='store_true',help="generate test features")
parser.add_argument("--train", action='store_true', help="train dense layers")
parser.add_argument("--predict", action='store_true', help="predict test data and save")
parser.add_argument("--sub", nargs=2, help="generate submission file")
parser.add_argument("--showconv", action='store_true', help="show summary of conv model")

args = parser.parse_args()
if args.gentrain:
    print('generating train features...')
    gen_train_features()
    print('done')
if args.train:
    print('training dense layer...')
    #train()
    train_conv_model()
    print('done')

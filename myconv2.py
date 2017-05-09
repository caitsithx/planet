import numpy as np 
import pandas as pd 
import os, gc, bcolz, glob

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D

import cv2
from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time
import argparse

DATA_DIR = '/home/chicm/data/planet'
RESULT_DIR = '/home/chicm/data/planet/results'
PREDICTS_FILE = RESULT_DIR + '/preds'

img_size = (256,256)
batch_size = 128

df_train = pd.read_csv(DATA_DIR+'/train_clean.csv')
df_test = pd.read_csv(DATA_DIR+'/sample_submission.csv')

labels = ['haze', 'cultivation', 'blooming', 'partly_cloudy', 'habitation', 'primary', 
            'road', 'agriculture', 'selective_logging', 'artisinal_mine', 'slash_burn', 
            'blow_down', 'cloudy', 'bare_ground', 'conventional_mine', 'clear', 'water']

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
def load_array(fname):
    return bcolz.open(fname)[:]

def gen_train_data():
    x_train = []   
    y_train = []

    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}
    print("reading images...")
    for f, tags in tqdm(df_train.values, miniters=5000):
        fn = DATA_DIR+'/train-jpg/'+f+'.jpg'
        img = cv2.imread(fn) 
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, img_size))
        y_train.append(targets)

    print("converting to ndarray...")
    x_train = np.array(x_train, np.float32) / 255.
    y_train = np.array(y_train, np.uint8)
    
    return x_train, y_train

def gen_test_data():
    x_test = []
    for f, tags in tqdm(df_test.values, miniters=5000):
        fn = DATA_DIR+'/test-jpg/'+f+'.jpg'
        img = cv2.imread(fn)
        x_test.append(cv2.resize(img, img_size))
    x_test = np.array(x_test, np.float32) / 255.
    return x_test


#vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
#def vgg_preprocess(x):
#    x = x - vgg_mean
#    return x[:, ::-1] # reverse axis rgb->bgr

def get_conv_layers(input_shape):
    return [
        #Lambda(vgg_preprocess, input_shape=input_shape, output_shape=input_shape),    
        Conv2D(16, (3, 3), activation='relu', input_shape = input_shape),
        BatchNormalization(axis=1),
        Conv2D(16, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),

        Dropout(0.15),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(17, activation='sigmoid')
    ]

def get_conv_model(input_shape):
    model = Sequential(get_conv_layers(input_shape))
    model.compile(Adam(lr=0.001, decay=0.05), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_conv_model(): 
    x_train, y_train = gen_train_data()
    nfolds = 4
    num_fold = 0
    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
    print("start training...")
    for train_index, val_index in kf:     
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        
        X_val = x_train[val_index]
        Y_val = y_train[val_index]

        num_fold += 1
        kfold_weights_path = RESULT_DIR+'/kfw2_'+str(num_fold)+'.h5'

        print('Start KFold{}'.format(num_fold))

        model = get_conv_model(x_train.shape[1:])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, verbose=0),
            ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_val, Y_val), epochs=100, 
            callbacks=callbacks, shuffle=True)

def ensemble():
    preds = []
    test = gen_test_data()
    w_files = glob.glob(RESULT_DIR+'/kfw2_*.h5')
    for fn in w_files:
        model = get_conv_model(test.shape[1:])
        print(fn)
        model.load_weights(fn)
        preds.append(model.predict(test, batch_size=128))
    m = np.mean(preds, axis=0)
    print(m.shape)
    save_array(PREDICTS_FILE, m)

def submit(filename, min_val=0.18):
    result = load_array(PREDICTS_FILE)
    print(result[:20])
    result = pd.DataFrame(result, columns=labels)

    preds = []
    for i in tqdm(range(result.shape[0]), miniters=1000):
        a = result.ix[[i]]
        a = a.apply(lambda x: x > min_val, axis=1)
        a = a.transpose()
        a = a.loc[a[i] == True]
        ' '.join(list(a.index))
        preds.append(' '.join(list(a.index)))
    df_test['tags'] = preds
    print(df_test)
    df_test.to_csv(RESULT_DIR+'/'+filename, index=False)

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="train dense layers")
parser.add_argument("--predict", action='store_true', help="predict test data and save")
parser.add_argument("--sub", nargs=2, help="generate submission file")
parser.add_argument("--showconv", action='store_true', help="show summary of conv model")

args = parser.parse_args()
if args.train:
    print('training dense layer...')
    train_conv_model()
    print('done')
if args.predict:
    print('predicting...')
    ensemble()
    print('done')
if args.sub:
    print('generating submision file...')
    submit(args.sub[0], (float)(args.sub[1]))
    print('done')
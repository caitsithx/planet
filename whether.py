import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2, gc
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D

DATA_DIR = '/home/chicm/data/planet'
RESULT_DIR = DATA_DIR+'/results'

df_train = pd.read_csv(DATA_DIR+'/train.csv')

def get_imgs(tag):
    img_fns = []
    for f, tags in df_train.values:
        if tag in tags.split(' '):
            img_fns.append(f)
    return img_fns

cloudy_fns = get_imgs('cloudy')
haze_fns = get_imgs('haze')
pcloudy_fns = get_imgs('partly_cloudy')
clear_fns = get_imgs('clear')

wt_labels = ['cloudy', 'partly_cloudy', 'haze', 'clear']
def one_hot(wt):
    if wt == 'cloudy':
        return [1, 0, 0, 0]
    elif wt == 'partly_cloudy':
        return [0, 1, 0, 0]
    elif wt == 'haze':
        return [0, 0, 1, 0]
    elif wt == 'clear':
        return [0, 0, 0, 1]
    else:
        pass
fns = cloudy_fns + pcloudy_fns + haze_fns + clear_fns[:10000]

fn_map = {k:'cloudy' for k in cloudy_fns}
map2 = {k:'partly_cloudy' for k in pcloudy_fns}
map3 = {k:'haze' for k in haze_fns}
map4 = {k:'clear' for k in clear_fns[:10000]}
fn_map.update(map2)
fn_map.update(map3)
fn_map.update(map4)
len(fn_map)

rfns = np.random.permutation(fns)

x_train = []
y_train = []
for f in tqdm(rfns):
    fn = DATA_DIR + '/train-jpg/'+f+'.jpg'
    img = cv2.imread(fn)
    x_train.append(cv2.resize(img, (128,128)))
    y_train.append(one_hot(fn_map[f]))
x_train = np.array(x_train, np.float32) / 255.
y_train = np.array(y_train, np.uint8)
print(x_train.shape)
print(y_train.shape)


def get_conv_layers(input_shape):
    return [    
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.15),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(axis=1),
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.15),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
        #Conv2D(4, (3,3), activation='relu'),
        #BatchNormalization(axis=1),
        #GlobalAveragePooling2D(),
        #Activation('softmax')
    ]

def get_conv_model(input_shape):
    model = Sequential(get_conv_layers(input_shape))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

batch_size = 64
model = get_conv_model(x_train.shape[1:])

split = 20000
X_train = x_train[:split]
Y_train = y_train[:split]
X_val = x_train[split:]
Y_val = y_train[split:]

model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs = 4, validation_data=(X_val, Y_val))

model.optimizer.lr = 0.01
model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs = 4, validation_data=(X_val, Y_val))

model.optimizer.lr = 0.0001
model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs = 10, validation_data=(X_val, Y_val))
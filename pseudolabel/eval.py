import settings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os, glob
import cv2
import random
import argparse
import bcolz
import pandas as pd
import random
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import fbeta_score
from utils import save_array, load_array, get_acc_from_w_filename
from utils import get_multi_classes, create_model, optimise_f2_thresholds
from utils import voting
from cscreendataset import classes, label_map

data_dir = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
RESULT_DIR = data_dir + '/results/92952'

PRED_DIR = RESULT_DIR + '/preds'

TEST_LIST_FILE = data_dir + '/sample_submission_v2.csv'
THRESHOLD_FILE_ENS = RESULT_DIR + '/best_threshold_ens.dat'

PRED_FILE = RESULT_DIR + '/pred_ens.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'
PRED_LIST = RESULT_DIR + '/pred_list.dat'

PRED_VAL = RESULT_DIR + '/pred_val.dat'
PRED_VAL_RAW = RESULT_DIR + '/pred_val_raw.dat'
VAL_LABELS = RESULT_DIR + '/val_labels.dat'

PRED_WEATHER = RESULT_DIR + '/pred_weather.dat'

#classes = ['clear', 'haze', 'partly_cloudy', 'cloudy', 'primary', 'agriculture', 'water', 'cultivation', 'habitation', 'road',
#            'slash_burn', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']

weak_classes = ['agriculture', 'water', 'cultivation', 'habitation', 'road']

print(classes)

def get_pred_from_probs(probs, thr):
    preds = np.zeros(probs.shape)
    for i in range(len(thr)):
        preds[:, i] = (probs[:, i] > thr[i]).astype(np.int)
    return preds

def f_measure(preds, labels, beta=2):
    
    SMALL = 1e-6  # 0  #1e-12
    p = preds
    l = labels

    num_pos = np.sum(p,  1) + SMALL
    num_pos_hat = np.sum(l,  1)
    tp = np.sum(l * p, 1)
    precise = tp / num_pos
    recall = tp / num_pos_hat

    fs = (1 + beta * beta) * precise * recall / \
        (beta * beta * precise + recall + SMALL)
    f = np.sum(fs) / preds.shape[0]
    return f


def f2(x, y):
    #print(x.shape)
    #print(y.shape)
    
    #p2 = np.zeros_like(p)
    #for i in range(17):
    #  p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, x, beta=2, average='samples')
    return score

def calc_acc(pred, y):
    #print(y.shape[0])
    if pred.shape != y.shape:
        print('ERROR, shape incorrect:{},{}'.format(pred.shape, y.shape))
    return np.sum((pred == y).astype(np.int)) / y.shape[0] 

def get_false_positive(pred, y):
    pred = pred.astype(np.int)
    y = y.astype(np.int)
    t1 = (y == 0).astype(np.int)
    t2 = (pred == 1).astype(np.int)
    return np.sum(t1*t2) #, t1*t2

def get_true_negative(pred, y):
    pred = pred.astype(np.int)
    y = y.astype(np.int)
    t1 = (y == 1).astype(np.int)
    t2 = (pred == 0).astype(np.int)
    return np.sum(t1*t2) #, t1*t2

def eval_val():
    thr = load_array(THRESHOLD_FILE_ENS)
    print(thr)

    pred_val = load_array(PRED_VAL)
    val_labels = load_array(VAL_LABELS)
    pred_val = get_pred_from_probs(pred_val, thr)
    
    print(np.sum(val_labels[0], 0))

    scores1 = []
    #scores2 = []
    fp = []
    tn = []
    for i in range(len(thr)):
        p = pred_val[:, i]
        y = val_labels[0][:, i]
        #scores1.append(f2(p, y))
        #scores2.append(f_measure(p, y))
        scores1.append(calc_acc(p, y))
        fp.append(get_false_positive(p, y))
        tn.append(get_true_negative(p,y))
    
    print(scores1)
    #print(scores2)
    print('true negative:')
    print(tn)
    print('false positive:')
    print(fp)

    score1 = f2(pred_val, val_labels[0])
    score2 = f_measure(pred_val, val_labels[0])
    print(score1)
    print(score2)

def stats():
    csv = pd.read_csv(data_dir + '/train_v2.csv') 
    ser = csv['tags'].map(lambda x : x.split()) 
    tags = [] 
    for s in ser:     
        tags = tags + s 
    ser = pd.Series(tags) 
    print(ser.head())
    print(ser.value_counts())

def create_train_water_split():
    df = pd.read_csv(data_dir + '/train_v2.csv')
    print(df.shape)
    df1 = df[df['tags'].apply(lambda x: 'road' in x.split(' ') or 'water' in x.split(' ')
            or 'agriculture' in x.split(' ') or 'cultivation' in x.split(' ') or 'habitation' in x.split(' '))]
    print(df1.shape)
    df2 = df[df['tags'].apply(lambda x: 'road' not in x.split(' ') and 'water' not in x.split(' ')
            and 'agriculture' not in x.split(' ') and 'cultivation' not in x.split(' ') and 'habitation' not in x.split(' '))]
    print(df2.shape)

    for c in weak_classes:
        df_c = df[df['tags'].apply(lambda x: c in x.split(' '))]
        print('{}:{}'.format(c, df_c.shape[0]))

    df_cult = df[df['tags'].apply(lambda x: 'cultivation' in x.split(' '))]
    print(df_cult.shape)
    df_hab = df[df['tags'].apply(lambda x: 'habitation' in x.split(' '))]
    print(df_hab.shape)

    df_res = pd.concat([df1, df_cult, df_hab], axis=0, ignore_index=True)
    print(df_res.shape)
    print(df_res.head())
    df_res = df_res.reindex(np.random.permutation(df_res.index))
    print(df_res.shape)
    print(df_res.head())

    # remove other tags
    df_res['tags'] = df_res['tags'].apply(lambda x: ' '.join([item for item in x.split(' ') if item in weak_classes]))
    print(df_res.head())
    df_res.to_csv(data_dir+'/train_water.csv', index=False)

def submit(filename):
    df_test = pd.read_csv(TEST_LIST_FILE)
    #preds = load_array(PRED_FILE)
    #preds = load_array(PRED_WEATHER)
    preds = get_final_preds()
    preds = np.mean(preds, axis=0)
    
    threshold = load_array(THRESHOLD_FILE_ENS).tolist()
    #threshold = 0.18
    print(threshold)
    
    for i, pred in enumerate(preds):
        tags = get_multi_classes(pred, classes, threshold)
        df_test['tags'][i] = tags

    df_test.to_csv(RESULT_DIR+'/'+filename, index=False)
    print(df_test.head())


def create_sub_from_pred_file():
    threshold = load_array(THRESHOLD_FILE_ENS).tolist()
    print(threshold)
    filenames = glob.glob(PRED_DIR + '/*/final')
    for f in filenames:
        print(f)
        probs = load_array(f)
        df_test = pd.read_csv(TEST_LIST_FILE)
        for i, pred in enumerate(probs):
            tags = get_multi_classes(pred, classes, threshold)
            df_test['tags'][i] = tags
        df_test.to_csv(f+'_sub.csv', index=False)

def create_sub_from_weighted_model(model_file_names, weights, sub_filename):
    threshold = load_array(THRESHOLD_FILE_ENS).tolist()
    print(threshold)
    preds = None
    total_weight = 0
    for i, fn in enumerate(model_file_names):
        p = load_array(PRED_DIR + '/' + fn + '/final') * weights[i]
        if preds is None:
            preds = p
        preds += p
        total_weight += weights[i]

    preds /= total_weight
    df_test = pd.read_csv(TEST_LIST_FILE)

    for i, pred in enumerate(preds):
        tags = get_multi_classes(pred, classes, threshold)
        df_test['tags'][i] = tags
    df_test.to_csv(RESULT_DIR + '/' + sub_filename, index=False)
    


parser = argparse.ArgumentParser()
parser.add_argument("--stats", action='store_true', help="ensemble predict")
parser.add_argument("--predf", action='store_true', help="ensemble predict")
parser.add_argument("--ensval", action='store_true', help="ensemble predict")
parser.add_argument("--thr", action='store_true', help="ensemble predict")
parser.add_argument("--weather", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=1, help="generate submission file")
parser.add_argument("--vote", nargs=1, help="generate submission file")
parser.add_argument("--create", action='store_true', help="ensemble predict")
parser.add_argument("--eval", action='store_true', help="generate submission file")

args = parser.parse_args()
if args.stats:
    stats()
    print('done')
if args.sub:
    print('generating submision file...')
    submit(args.sub[0])
    print('done')
    print('Please find submisson file at: {}'.format(RESULT_DIR+'/'+args.sub[0]))
if args.vote:
    print('generating voting submision file...')
    create_voting_sub(args.vote[0])
    print('done')
if args.create:
    create_train_water_split()
if args.predf:
    mfiles = ['dense169_11_0.93117_.pth', 'vgg16bn_10_0.92676_.pth', 'vgg19bn_14_0.92884_.pth']
    for mfile in mfiles:
        make_preds(mfile)
if args.eval:
    eval_val()

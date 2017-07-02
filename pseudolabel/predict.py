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
from cscreendataset import classes, get_test_loader, get_val_loader

data_dir = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
RESULT_DIR = settings.RESULT_DIR

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

water_threshold = [0.54, 0.35, 0.46,  0.49,  0.49] #[0.34, 0.22, 0.29, 0.29, 0.29]
WATER_PRED = '/home/chicm/data/planet/results/93091_water/water/pred_ens.dat'

batch_size = 16

w_file_matcher = ['dense161*pth', 'dense201*pth','dense169*pth','dense121*pth','inceptionv3*pth',
    'res50*pth','res101*pth', 'res152*pth', 'vgg16*pth', 'vgg19*pth']

if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
if not os.path.exists(PRED_DIR):
    os.mkdir(PRED_DIR)

def make_preds_val(net):
    #loader = get_test_loader()
    val_loader = get_val_loader(net, shuffle=False)
    preds = []
    y = []
    net.eval()
    for iter, (images, labels, fn) in enumerate(val_loader, 0):
        images = Variable(images.cuda())
        logits = net(images)
        probs = F.sigmoid(logits).data.cpu().tolist()
        for p in probs:
            preds.append(p)
        for label in labels.tolist():
            y.append(label)
    y = np.array(y)
    preds = np.array(preds)

    return preds, y

def ensemble_val_data():
    preds_raw = []
    labels = []
    
    for match_str in w_file_matcher:
        os.chdir(MODEL_DIR)
        w_files = glob.glob(match_str)
        for w_file in w_files:
            full_w_file = MODEL_DIR + '/' + w_file
            mname = w_file.split('_')[0]
            print(full_w_file)
            model = create_model(mname)
            model.load_state_dict(torch.load(full_w_file))

            pred,y = make_preds_val(model)
            #pred = np.array(pred)
            preds_raw.append(pred)
            labels.append(y)
            del model

    save_array(PRED_VAL_RAW, preds_raw)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_VAL, preds)
    save_array(VAL_LABELS, labels)
    return preds, labels

def find_best_threshold():
    preds = load_array(PRED_VAL)
    labels = load_array(VAL_LABELS)
    print(np.array(labels).shape)
    for i in range(1, len(labels)):
        for j in range(len(labels[i])):
            for k in range(len(labels[i][j])):
                if labels[i][j][k] != labels[i-1][j][k]:
                    print('error, check labels failed')
                    exit()
    
    x = optimise_f2_thresholds(labels[0], preds)
    print('best threshold:')
    print(x)
    save_array(THRESHOLD_FILE_ENS, x)

def get_one_weather(weather, thr, dominate, delta = 0.2):
    res = []
    for wrow in weather:
        row = wrow.copy()
        maxw = 0
        maxindex = -1
        for i, w in enumerate(row):
            if w > maxw:
                maxw = w
                maxindex = i
        if maxw <= thr[maxindex]:
            row = [0]*4
            row[maxindex] = 0.99
        if maxw >= dominate:
            for j in range(4):
                if row[j] <=  maxw - delta:
                    row[j] = 0
        res.append(row)

    return np.array(res)

def force_one_weather(weather, thr):
    res = []
    for wrow in weather:
        row = wrow.copy()
        maxw = 0
        maxindex = -1
        for i, w in enumerate(row):
            if w-thr[i] > maxw:
                maxw = w-thr[i]
                maxindex = i

        for j in range(4):
            if j == maxindex:
                row[j] = 0.99
            else:
                row[j] = 0
        res.append(row)

    return np.array(res)
            

def find_best_weather():
    thr = load_array(THRESHOLD_FILE_ENS)
    labels = load_array(VAL_LABELS)
    preds = load_array(PRED_VAL)

    print(labels.shape)
    weather = preds[:,0:4]
    y = labels[0, :, 0:4]
    print(y.shape)
    print(weather.shape)
    thr = thr[0:4]

    def mf(p):
        p2 = np.zeros_like(p)
        for i in range(4):
            p2[:, i] = (p[:, i] > thr[i]).astype(np.int)
        score1 = fbeta_score(y, p2, beta=2, average='samples')
        return score1

    base_score = mf(weather) #fbeta_score(y, weather, beta=2, average='samples')
    print('base score:{}'.format(base_score))
    max_score = base_score
    d = 0.5
    best_d = 0.5
    best_w = weather
    while d < 1:
        w = get_one_weather(weather, thr, d)
        score = mf(w)  #fbeta_score(y, w, beta=2, average='samples')
        print('score{}, d:{}'.format(score, d))
        if score > max_score:
            max_score = score
            best_d = d
            best_w = w
        d += 0.1
    
    print('best d:{}'.format(best_d))
    w1 = force_one_weather(weather, thr)
    score1 = mf(w1)
    print('force one weather score:{}'.format(score1))

    if max_score > base_score+0.00001:
        test_preds = load_array(PRED_FILE)
        test_w = test_preds[:, 0:4]
        w = get_one_weather(test_w, thr, best_d)
        test_preds[:, 0:4] = w
        #preds[:, 0:4] = best_w

        save_array(PRED_WEATHER, test_preds)

def make_preds(weight_file_name):
    full_w_file = MODEL_DIR + '/' + weight_file_name

    mname = weight_file_name.split('_')[0]
    print(full_w_file)
    net = create_model(mname)
    net.load_state_dict(torch.load(full_w_file))
    
    pred_dir = os.path.join(PRED_DIR, weight_file_name)

    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    all_preds = []

    for r in range(4):
        print('Rotation {}'.format(r))
        loader = get_test_loader(net, rotate=r)
        preds = []
        net.eval()
        for i, (img, _) in enumerate(loader, 0):
            inputs = Variable(img.cuda())
            outputs = net(inputs)
            probs = F.sigmoid(outputs).data.cpu().tolist()
            for p in probs:
                preds.append(p)
        preds = np.array(preds)
        print(preds[:5])
        save_array(pred_dir+'/roate'+str(r), preds)
        all_preds.append(preds)
    mean_preds = np.mean(all_preds, axis=0)
    save_array(pred_dir+'/final', mean_preds)

    del net

    return mean_preds

def ensemble():
    preds_raw = []
    model_file_list = []
    for match_str in w_file_matcher:
        os.chdir(MODEL_DIR)
        w_files = glob.glob(match_str)
        for w_file in w_files:
            model_file_list.append(w_file)

            pred = make_preds(w_file)
            pred = np.array(pred)
            preds_raw.append(pred)

    save_array(PRED_LIST, model_file_list)
    save_array(PRED_FILE_RAW, preds_raw)
    preds = np.mean(preds_raw, axis=0)
    save_array(PRED_FILE, preds)

def get_final_preds(single=True):
    if single:
        filenames = glob.glob(PRED_DIR + '/*/final')
    else:
        filenames = glob.glob(PRED_DIR + '/*/roate*')
    preds = []
    for f in filenames:
        print(f)
        pred = load_array(f)
        preds.append(pred)
    return preds

def create_voting_sub(fn):
    threshold = load_array(THRESHOLD_FILE_ENS).tolist()
    print(threshold)
    preds = get_final_preds(False)
    
    print(len(preds))
    voted = voting(preds, threshold)
    print(voted[:5])

    df_test = pd.read_csv(TEST_LIST_FILE)

    for i, pred in enumerate(voted):
        tags = get_multi_classes(pred, classes, threshold)
        df_test['tags'][i] = tags
    df_test.to_csv(RESULT_DIR+'/'+fn, index=False)


def submit(filename):
    df_test = pd.read_csv(TEST_LIST_FILE)
    #preds = load_array(PRED_FILE)
    #preds = load_array(PRED_WEATHER)
    preds = get_final_preds()
    preds = np.mean(preds, axis=0)

    threshold = load_array(THRESHOLD_FILE_ENS).tolist()
    #threshold = 0.18
    print(threshold)

    if False:
        threshold[5:10] = water_threshold
        print(threshold)
        print(preds[:1])
        print(load_array(WATER_PRED).shape)
        preds[:, 5:10] = load_array(WATER_PRED)
        print(preds[:1])
        #return
    
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
parser.add_argument("--ens", action='store_true', help="ensemble predict")
parser.add_argument("--predf", action='store_true', help="ensemble predict")
parser.add_argument("--ensval", action='store_true', help="ensemble predict")
parser.add_argument("--thr", action='store_true', help="ensemble predict")
parser.add_argument("--weather", action='store_true', help="ensemble predict")
parser.add_argument("--sub", nargs=1, help="generate submission file")
parser.add_argument("--vote", nargs=1, help="generate submission file")
parser.add_argument("--subsingle", action='store_true', help="ensemble predict")
parser.add_argument("--subw", nargs=1, help="generate submission file")

args = parser.parse_args()
if args.ens:
    ensemble()
    print('done')
if args.ensval:
    ensemble_val_data()
if args.thr:
    find_best_threshold()
if args.weather:
    find_best_weather()
if args.sub:
    print('generating submision file...')
    submit(args.sub[0])
    print('done')
    print('Please find submisson file at: {}'.format(RESULT_DIR+'/'+args.sub[0]))
if args.vote:
    print('generating voting submision file...')
    create_voting_sub(args.vote[0])
    print('done')
if args.subsingle:
    create_sub_from_pred_file()
if args.subw:
    mfiles = ['dense161_8_0.93172_.pth', 'dense161_5_0.93093_.pth', 'dense201_12_0.92993_.pth', 'dense121_11_0.93093_.pth', 
        'res101_14_0.93156_.pth', 'res101_8_0.93120_.pth', 'res152_6_0.93012_.pth', 'res152_5_0.92974_.pth', 'inceptionv3_12_0.92727_.pth', 
        'vgg19bn_3_0.92535_.pth', 'dense169_11_0.92701_.pth']
    w = [1.2, 1.2, 1.1, 1, 1.1, 1.2, 1.1, 1.1, 0.8, 0.8, 0.8]
    create_sub_from_weighted_model(mfiles, w, args.subw[0])
    print('done')
if args.predf:
    mfiles = ['dense161_28_0.93236_.pth']
    for mfile in mfiles:
        make_preds(mfile)

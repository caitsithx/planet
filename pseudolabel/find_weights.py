
import settings
import os,glob
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import fbeta_score
from scipy.optimize import minimize
from utils import load_array, save_array


data_dir = settings.DATA_DIR
MODEL_DIR = settings.MODEL_DIR
RESULT_DIR = data_dir + '/results'

THRESHOLD_FILE_ENS = RESULT_DIR + '/best_threshold_ens.dat'

PRED_FILE = RESULT_DIR + '/pred_ens.dat'
PRED_FILE_RAW = RESULT_DIR + '/pred_ens_raw.dat'

PRED_VAL_RAW = RESULT_DIR + '/pred_val_raw.dat'
VAL_LABELS = RESULT_DIR + '/val_labels.dat'

preds = load_array(PRED_VAL_RAW)
val_labels = load_array(VAL_LABELS)
y = val_labels[0]
thr = load_array(THRESHOLD_FILE_ENS)

def mf(weights):
    
    p_w = np.zeros_like(preds[0])
    total_weights = 0
    for i, pred in enumerate(preds):
        p_w += pred * weights[i]
        total_weights += weights[i]
    p_w = p_w / total_weights

    p2 = np.zeros_like(p_w)
    for i in range(17):
        p2[:, i] = (p_w[:, i] > thr[i]).astype(np.int)
    #p2 = p_w > thr

    score1 = fbeta_score(y, p2, beta=2, average='samples')
    return 1- score1

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros(preds[0].shape)
    for weight, prediction in zip(weights, preds):
            final_prediction += weight*prediction
    loss = log_loss(labels, final_prediction)
    return loss

#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
def calc():
    starting_values = [0.5]*len(preds)
    #starting_values = np.random.rand(len(preds))


    #adding constraints  and a different solver as suggested by user 16universe
    #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    #our weights are bound between 0 and 1
    bounds = [(0,1)]*len(preds)

    res = minimize(mf, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    return res['fun'], res['x']

best = 100
best_w = None
for i in range(50):
    loss, weights = calc()
    if loss < best:
        best = loss
        best_w = weights

print('best loss:{}'.format(best))
print('best weights:{}'.format(best_w))

#weight_list = [(w_filenames[i], best_w[i]) for i in range(len(w_filenames))]
#print(weight_list)

#save_array(WEIGHTS_FILE, weight_list)
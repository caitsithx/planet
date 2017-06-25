import numpy as np
import pandas as pd
from cscreendataset import classes, label_map
import settings

RESULT_DIR = settings.DATA_DIR + '/results'

tgt_file = RESULT_DIR+'/sub1-pl-cult.csv'

df_train = pd.read_csv(RESULT_DIR+'/sub1-pl.csv')

for i, row in enumerate(df_train.values):
    tags = row[1].split(' ')
    if 'cultivation' in tags:
        if not 'agriculture' in tags:
            tags.append('agriculture')
    df_train.ix[i, 'tags'] = ' '.join(tags)

df_train.to_csv(tgt_file, index=False)

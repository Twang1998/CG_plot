import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from sklearn import preprocessing
# import seaborn as sns
df = pd.read_csv('CG_QA_test.csv')

all_score_path = 'Score_collation'
score_files = os.listdir(all_score_path)
for score_file in score_files:
    cur_df = pd.read_csv(os.path.join(all_score_path,score_file))
    # print(len(cur_df))

    # standard = preprocessing.StandardScaler().fit_transform(np.array(cur_df['Score']).reshape(-1, 1))
    # norm = standard.reshape(-1,)
    # norm = (norm+3)*100/6
    # norm[norm<0] = 0
    # norm[norm>100] = 100
    # norm = np.round(norm,2)
    # df[score_file.split('.')[0]] = norm

    df[score_file.split('.')[0]] = np.array(cur_df['Score'])


img_list = np.array(df['Image'])
MOS = []
var = []
for i in range(len(img_list)):
    single_score = np.array(df.iloc[i,2:])
    MOS.append(np.round(np.mean(single_score),2))
    var.append(np.round(np.var(single_score),2))
df['MOS'] = MOS
df['var'] = var
df.to_csv('MOS_var_test_no_filter.csv',index=False)
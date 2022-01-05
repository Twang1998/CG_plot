import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

def cal_correlation(a,b):
    srcc = stats.spearmanr(a,b)[0]
    plcc = stats.pearsonr(a,b)[0]
    krcc = stats.kendalltau(a,b)[0]

    print('srcc: {:.4f}, plcc: {:.4f}, krcc: {:.4f}'.format(srcc,plcc,krcc))
    return srcc,plcc,krcc

df = pd.read_csv('MOS_var_test_just_norm.csv')

# ## 两两之间
# data = np.array(df.iloc[:,2:-2])
# print(data.shape)

# result = np.ones((data.shape[1],data.shape[1]))

# for i in range(data.shape[1]-1):
#     for j in range(i+1,data.shape[1]):
#         srcc,plcc,krcc = cal_correlation(data[:,i],data[:,j])
#         result[i][j] = np.round(srcc,4)
#         result[j][i] = np.round(srcc,4)
# print(result)
# result_df = pd.DataFrame(result)
# result_df.to_csv('temp.csv',index=False)


### 所有人和MOS
data = np.array(df.iloc[:,2:-1])
print(data.shape)

for i in range(data.shape[1]-1):
    srcc,plcc,krcc = cal_correlation(data[:,i],data[:,-1])
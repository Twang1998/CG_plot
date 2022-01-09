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

# df = pd.read_csv('MOS_var_test_just_norm.csv')

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


# ### 所有人和MOS
# data = np.array(df.iloc[:,2:-1])
# print(data.shape)

# for i in range(data.shape[1]-1):
#     srcc,plcc,krcc = cal_correlation(data[:,i],data[:,-1])


## ---------------------------------------------------###
## 看一看异常值剔除前后的srcc对比，以及缩放到[0:100]和直接平均的srcc
df1 = pd.read_csv('MOS_var_test_filter.csv')
df2 = pd.read_csv('MOS_var_test_norm_filter.csv')
df3 = pd.read_csv('MOS_var_test_no_filter.csv')
df4 = pd.read_csv('MOS_var_test_norm_no_filter.csv')

a = np.array(df1['MOS'])
b = np.array(df2['MOS'])
c = np.array(df3['MOS'])
d = np.array(df4['MOS'])

cal_correlation(a,b)
cal_correlation(a,c)
cal_correlation(a,d)

# srcc: 0.9990, plcc: 0.9991, krcc: 0.9753
# srcc: 0.9937, plcc: 0.9942, krcc: 0.9384
# srcc: 0.9938, plcc: 0.9942, krcc: 0.9348

## -------------------------------------##
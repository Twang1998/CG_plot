import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
plt.rc('font',family='Times New Roman')
# plt.style.use('fivethirtyeight')
# plt.style.use(['science', 'no-latex'])
# 自定义函数形式
def func(x, beta1, beta2, beta3, beta4, beta5):
    yhat = beta1 * (0.5 - np.divide(1,(1+np.exp(beta2*(x-beta3))))) + beta4*x + beta5

    return yhat

def fit_function(y_label, y_output):
    # beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(func, y_output, y_label, p0=None, maxfev=100000000)
    y_output_logistic = func(y_output, *popt)
    plt.plot(np.arange(0,5,0.1),func(np.arange(0,5,0.1),*popt))
    plt.show()
    
    return y_output_logistic



df = pd.read_csv('MOS_var_test.csv')
y_label = np.array(df['wangtao'])
y_output = np.array(df['wangtao2'])

a = y_label
b = y_output
b = fit_function(y_label,y_output)
srcc = stats.spearmanr(a,b)[0]
plcc = stats.pearsonr(a,b)[0]
krcc = stats.kendalltau(a,b)[0]

print('srcc: {:.4f}, plcc: {:.4f}, krcc: {:.4f}'.format(srcc,plcc,krcc))
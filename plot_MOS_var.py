from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
plt.rc('font',family='Times New Roman')
# plt.style.use('fivethirtyeight')
# plt.style.use(['science', 'no-latex'])
# 自定义函数形式
def func(x, a):
    return a * x * (5-x)



df = pd.read_csv('MOS_var_test_just.csv')
df = df.sort_values(by="MOS",ascending=True)
mos_list = np.array(df['MOS'])
var_list = np.array(df['var'])

popt, pcov = curve_fit(func, mos_list, var_list)

a = popt[0]
print(a)
yvals = func(np.arange(0,5,0.01), a)
# distplot 简版就是hist 加上一根density curve

fig = plt.figure(figsize=(12,8))
plt.scatter(mos_list,var_list,s=10,c='g',alpha=0.5)
plt.plot(np.arange(0,5,0.01),yvals,lw=3,label='fit curve, a = {}'.format(np.round(a,2)))
plt.xlabel('MOS')
plt.ylabel('Variance')
plt.xlim(0,5)
plt.ylim(-0.1,2)

# plt.title('Probability density of MOS')
plt.legend()
# plt.savefig('MOS_var_dist.png',dpi = 500,bbox_inches = 'tight',transparent = True)
plt.show()
    

from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

df = pd.read_csv('feature.csv')

#'luminance','contrast','color','SI'
luminance = np.array(df['luminance'])
contrast = np.array(df['contrast'])
color = np.array(df['color'])
SI = np.array(df['SI'])

luminance = (luminance-np.min(luminance))/(np.max(luminance)-np.min(luminance))
# print(luminance)
contrast = (contrast-np.min(contrast))/(np.max(contrast)-np.min(contrast))
color = (color-np.min(color))/(np.max(color)-np.min(color))
SI = (SI-np.min(SI))/(np.max(SI)-np.min(SI))

fig = plt.figure(figsize=(12,8))

# # 绘制男女乘客年龄的核密度图
# sns.distplot(Age_Male, hist = False, kde_kws = {'color':'red', 'linestyle':'-'},
#              norm_hist = True, label = '男性')
# # 绘制女性年龄的核密度图
# sns.distplot(Age_Female, hist = False, kde_kws = {'color':'black', 'linestyle':'--'},
#              norm_hist = True, label = '女性')


sns.kdeplot(luminance, shade=True,lw = 3,label = 'luminance')
sns.kdeplot(contrast, shade=True,lw = 3,label = 'contrast')
sns.kdeplot(color, shade=True,lw = 3,label = 'color')
sns.kdeplot(SI, shade=True,lw = 3,label = 'SI')
# sns.distplot(contrast,hist = True, bins = 30,kde_kws={"lw":3}, norm_hist = True, label = 'contrast')
# sns.distplot(color,hist = True, bins = 30,kde_kws={"lw":3}, norm_hist = True, label = 'color')
# sns.distplot(SI,hist = True, bins = 30,kde_kws={"lw":3}, norm_hist = True, label = 'SI')
plt.xlabel('Values')
plt.ylabel('Density')
plt.xlim(0,1)
# plt.ylim(-0.1,2)

# plt.title('Probability density of MOS')
plt.legend()
plt.savefig('features.png',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()
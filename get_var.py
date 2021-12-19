import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('MOS_test.csv')
img_list = np.array(df['MOS'])

var = []
for i in range(len(img_list)):
    single_score = np.array(df.iloc[i,2:6])
    var.append(np.var(single_score))
df['var'] = var
df.to_csv('MOS_var_test.csv',index=False)
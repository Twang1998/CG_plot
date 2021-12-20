from typing import Collection
import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from collections import Counter 

df = pd.read_csv('CG_QA_test.csv')
img_list = np.array(df['Image'])

img_path = r'C:\Users\37151\Desktop\CGQA\CG_QA'
# width = []
# height = []
res = []

for img in img_list:
    tmp = cv2.imread(os.path.join(img_path,img))
    shape = tmp.shape
    # width.append(shape[1])
    # height.append(shape[0])
    res.append((shape[1],shape[0]))

count = Counter(res)
width_x = []
height_y = []
size = []

for key in list(count.keys()):
    width_x.append(key[0])
    height_y.append(key[1])
    size.append(count[key])

fig = plt.figure(figsize=(12,8))
plt.scatter(width_x,height_y,s = np.array(size)*5,c = size,alpha=1)
plt.colorbar()  # 显示颜色条
plt.xlabel('Width')
plt.ylabel('Height')
plt.savefig('resolution.png',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()
 
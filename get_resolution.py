import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt

df = pd.read_csv('CG_QA_test.csv')
img_list = np.array(df['Image'])

img_path = r'C:\Users\37151\Desktop\CGQA\CG_QA'
width = []
height = []

for img in img_list:
    tmp = cv2.imread(os.path.join(img_path,img))
    shape = tmp.shape
    width.append(shape[1])
    height.append(shape[0])

fig = plt.figure(figsize=(16,8))
plt.scatter(width,height)
plt.savefig('resolution.png',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()
 
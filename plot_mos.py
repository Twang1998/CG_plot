import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['science', 'no-latex'])
df = pd.read_csv('MOS_test.csv')
img_list = np.array(df['MOS'])

# distplot 简版就是hist 加上一根density curve
fig = plt.figure(figsize=(16,8))
sns.set_palette("hls")
# plt.rc("figure", figsize=(9, 5))
sns.distplot(img_list,kde_kws={"color": "seagreen", "lw":3, "label" : "Kernel Density Estimation" }, 
             hist_kws={"histtype": "stepfilled", "color": "slategray" })
plt.xlabel('MOS')
plt.ylabel('Probability density')
plt.xlim(0,5)
plt.title('Probability density of MOS')
plt.legend()
plt.savefig('MOS_dist.png',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()
 

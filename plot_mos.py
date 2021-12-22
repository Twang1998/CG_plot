import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font',family='Times New Roman',size=20)
# plt.style.use(['science', 'no-latex'])
df = pd.read_csv('MOS_test.csv')
img_list = np.array(df['Image'])
MOS_list = np.array(df['MOS'])
game_MOS = []
movie_MOS = []
for i,img in enumerate(img_list):
    if img[0] == 'g':
        game_MOS.append(MOS_list[i])
    else:
        movie_MOS.append(MOS_list[i])

# #### together
# # distplot 简版就是hist 加上一根density curve
# fig = plt.figure(figsize=(16,8))
# sns.set_palette("hls")
# # plt.rc("figure", figsize=(9, 5))
# sns.distplot(MOS_list,kde_kws={"color": "seagreen", "lw":3, "label" : "Kernel Density Estimation" }, 
#              hist_kws={"histtype": "stepfilled", "color": "slategray" })
# plt.xlabel('MOS')
# plt.ylabel('Probability density')
# plt.xlim(0,5)
# plt.title('Probability density of MOS')
# plt.legend(fancybox=False,frameon = False)
# plt.savefig('MOS_dist.png',dpi = 500,bbox_inches = 'tight',transparent = True)
# # plt.show()


#### seprate
#### together
# distplot 简版就是hist 加上一根density curve
fig = plt.figure(figsize=(16,8))
sns.set_palette("hls")
# plt.rc("figure", figsize=(9, 5))
sns.distplot(game_MOS,bins = 20,kde_kws={"color": "r", "lw":3, "label" : "Game" }, 
             hist_kws={"histtype": "stepfilled", "color": "tomato" ,"range": [0,5]})
sns.distplot(movie_MOS,bins = 20,kde_kws={"color": "b", "lw":3, "label" : "Movie" }, 
             hist_kws={"histtype": "stepfilled", "color": "royalblue","range": [0,5] })
# sns.distplot(MOS_list,kde_kws={"color": "seagreen", "lw":3, "label" : "Kernel Density Estimation" },hist=None)
plt.xlabel('MOS')
plt.ylabel('Probability density')
plt.xlim(0,5)
plt.title('Probability density of Game and Movie separately')
plt.legend(fancybox=False,frameon = False)
plt.savefig('MOS_game_movie_dist.pdf',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()

 

import pandas as pd
import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
plt.rc('font',family='Times New Roman',size=24)
# plt.style.use(['science', 'no-latex'])
df = pd.read_csv('MOS_var_test_norm_filter.csv')
MAX_SCORE = 100
img_list = np.array(df['Image'])
MOS_list = np.array(df['MOS'])
game_MOS = []
movie_MOS = []
for i,img in enumerate(img_list):
    if img[0] == 'g':
        game_MOS.append(MOS_list[i])
    else:
        movie_MOS.append(MOS_list[i])

### together
fig = plt.figure(figsize=(16,8))
plt.grid(ls='--',zorder=0) 
# sns.set_palette("hls")
# plt.rc("figure", figsize=(9, 5))
sns.histplot(MOS_list,bins = 20,kde=False,stat="density",binrange=(0,MAX_SCORE),color='lightskyblue',zorder=10)
sns.kdeplot(MOS_list,color='royalblue',lw = 3,label='Kernel Density Estimation',zorder=30)

# sns.distplot(MOS_list,kde_kws={"color": "seagreen", "lw":3, "label" : "Kernel Density Estimation" },hist=None)
plt.xlabel('MOS')
plt.ylabel('Probability density')
plt.xlim(0,MAX_SCORE)
plt.title('Probability density of MOS')
plt.legend(fancybox=False,frameon = False)

plt.savefig('plot_pic\\MOS_dist.pdf',dpi = 500,bbox_inches = 'tight',transparent = True)
plt.show()


#### seprate
# distplot 简版就是hist 加上一根density curve

# fig = plt.figure(figsize=(16,8))
# plt.grid(ls='--',zorder=0) 
# # sns.set_palette("hls")
# # plt.rc("figure", figsize=(9, 5))
# sns.histplot(game_MOS,bins = 20,kde=False,stat="density",binrange=(0,MAX_SCORE),color='tomato',zorder=10)
# sns.kdeplot(game_MOS,color='r',lw = 3,label='Game',zorder=30)

# sns.histplot(movie_MOS,bins = 20,kde=False,stat="density",binrange=(0,MAX_SCORE),color='royalblue',zorder=20)
# sns.kdeplot(movie_MOS,color='b',lw = 3,label='Movie',zorder=40)

# # sns.distplot(MOS_list,kde_kws={"color": "seagreen", "lw":3, "label" : "Kernel Density Estimation" },hist=None)
# plt.xlabel('MOS')
# plt.ylabel('Probability density')
# plt.xlim(0,MAX_SCORE)
# plt.title('Probability density of Game and Movie separately')
# plt.legend(fancybox=False,frameon = False)

# plt.savefig('plot_pic\\MOS_game_movie_dist.pdf',dpi = 500,bbox_inches = 'tight',transparent = True)
# plt.show()

 

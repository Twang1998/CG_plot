# from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os 
import argparse

from pandas.io.parsers import TextParser

# def arg_parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video_path', type=str)

#     config = parser.parse_args()

#     return config

def color_frame(frame):
    ## frame : np.ndarray
    B, G, R =  frame[0],frame[1],frame[2]
    rg = R - G
    yb = 0.5*(R+G)-B
    variance_rg = np.var(rg)
    variance_yb = np.var(yb)
    mu_rg = np.mean(rg)
    mu_yb = np.mean(yb)

    return np.sqrt(variance_rg+variance_yb) + 0.3*np.sqrt(mu_rg**2 + mu_yb**2)

def luminance_frame(gray_frame):
    return np.mean(gray_frame)

def contrast_frame(gray_frame):
    return np.std(gray_frame)

def SI_frame(gray_frame):
    # compute gradients along the x and y axis, respectively
    gX = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1)
    # compute the gradient magnitude and orientation
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

    return np.std(magnitude)

if __name__ == "__main__":
    """

    How To Run: 
    Change the 'video_path' in line 109
    'video_path' means a path include all the videos.

    """
    # video_path = 'output.mp4'
    # feature = get_feature_video(video_path)
    # print(feature)

    # config = arg_parse()

    df = pd.DataFrame(columns=('Image','luminance','contrast','color','SI'))

    # total_feature = []

    img_path = r'C:\Users\37151\Desktop\CGQA\CG_QA'
    df_img = pd.read_csv('CG_QA_test.csv')
    img_list = np.array(df_img['Image'])
    for i, img_name in enumerate(img_list) :
        print(i)
        img = cv2.imread(os.path.join(img_path,img_name))
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        total_feature =[] 
        total_feature.append(img_name)
        total_feature.append(luminance_frame(img_gray))
        total_feature.append(contrast_frame(img_gray))
        total_feature.append(color_frame(img))
        total_feature.append(SI_frame(img_gray))
        # print(total_feature)
        
        df.loc[i] = total_feature
    # print(df)
    df.to_csv('feature.csv', index= False)

import numpy as np
import cv2
import pandas as pd
import os
from scipy import ndimage
from read_video import read_video


def color_feature(color_video):
    color_video = color_video.astype(np.float32)
    res = []
    feature = []
    for idx in range(len(color_video)):
        frame = color_video[idx, ]

        R, G, B = cv2.split(frame)
        rg = R - G
        yb = 0.5*(R+G)-B
        variance_rg = np.var(rg)
        variance_yb = np.var(yb)
        mu_rg = np.mean(rg)
        mu_yb = np.mean(yb)

        feature.append(np.sqrt(variance_rg+variance_yb) + 0.3*np.sqrt(mu_rg**2 + mu_yb**2))

    res.extend((np.mean(feature), np.std(feature)))
    return res


def luminance_feature(color_video):
    color_video = color_video.astype(np.float32)
    res = []
    feature = np.mean(color_video, axis=(1,2,3))

    res.extend((np.mean(feature), np.std(feature)))
    return res


def rms_contrast_feature(gray_video):
    gray_video = gray_video.astype(np.float32)
    res = []
    feature = np.std(gray_video, axis=(1,2))

    res.extend((np.mean(feature), np.std(feature)))
    return res


def SI_feature(gray_video):
    gray_video = gray_video.astype(np.float32)
    res = []
    feature = []

    for idx in range(len(gray_video)):
        gray_frame = gray_video[idx, ]
        gX = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1)
        # compute the gradient magnitude and orientation
        magnitude = np.sqrt((gX ** 2) + (gY ** 2))
        # orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
        feature.append(np.std(magnitude))

    res.append(np.max(feature))
    return res


def TI_frame(gray_video):
    gray_video = gray_video.astype(np.float32)
    res = []
    feature = []
    prev = gray_video[0, ]

    for idx in range(1, len(gray_video)):
        frame_diff = gray_video[idx, ] - prev
        feature.append(np.std(frame_diff))
        prev = gray_video[idx, ]

    res.append(np.max(feature))
    return res


def face_num_frame(gray_video):
    #detector = cv2.CascadeClassifier('D:\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    res = []
    feature = []
    for idx in range(len(gray_video)):
        feature.append(len(detector.detectMultiScale(gray_video[idx, ])))

    res.extend((np.mean(feature), np.std(feature)))
    return res


def get_gaussian_spatial(gray_video):
    gray_video = gray_video.astype(np.float32)
    res = []
    feature = []

    for idx in range(len(gray_video)):
        I = gray_video[idx]
        # X scale 1
        f = ndimage.filters.gaussian_filter1d(I, 1, axis=0, order=1, output=None, mode='constant', cval=0.0)
        Fx_1 = ndimage.filters.gaussian_filter1d(f, 3, axis=1, order=0, output=None, mode='constant', cval=0.0)
        #Fx_1 = 255*(Fx_1 - np.min(Fx_1))/(np.max(Fx_1))

        # X scale 2
        f = ndimage.filters.gaussian_filter1d(I, 2, axis=0, order=1, output=None, mode='constant', cval=0.0)
        Fx_2 = ndimage.filters.gaussian_filter1d(f, 6, axis=1, order=0, output=None, mode='constant', cval=0.0)

        # X scale 3
        f = ndimage.filters.gaussian_filter1d(I, 4, axis=0, order=1, output=None, mode='constant', cval=0.0)
        Fx_3 = ndimage.filters.gaussian_filter1d(f, 12, axis=1, order=0, output=None, mode='constant', cval=0.0)

        # Y scale 1
        f = ndimage.filters.gaussian_filter1d(I, 1, axis=1, order=1, output=None, mode='constant', cval=0.0)
        Fy_1 = ndimage.filters.gaussian_filter1d(f, 3, axis=0, order=0, output=None, mode='constant', cval=0.0)

        # Y scale 2
        f = ndimage.filters.gaussian_filter1d(I, 2, axis=1, order=1, output=None, mode='constant', cval=0.0)
        Fy_2 = ndimage.filters.gaussian_filter1d(f, 6, axis=0, order=0, output=None, mode='constant', cval=0.0)

        # Y scale 3
        f = ndimage.filters.gaussian_filter1d(I, 4, axis=1, order=1, output=None, mode='constant', cval=0.0)
        Fy_3 = ndimage.filters.gaussian_filter1d(f, 12, axis=0, order=0, output=None, mode='constant', cval=0.0)

        feature.append([  np.mean(abs(Fx_1)),
                      np.mean(abs(Fx_2)),
                      np.mean(abs(Fx_3)),
                      np.mean(abs(Fy_1)),
                      np.mean(abs(Fy_2)),
                      np.mean(abs(Fy_3))])

    res.extend(np.mean(feature, axis=0).tolist() + np.std(feature, axis=0).tolist())
    # print(res)
    return res


def get_gaussian_temporal(gray_video):
    gray_video = gray_video.astype(np.float32)
    [K,M,N] = gray_video.shape

    V1 = ndimage.filters.gaussian_filter1d(gray_video, 2, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)
    # print(V1.shape)  ## ==V.shape
    V2 = ndimage.filters.gaussian_filter1d(gray_video, 4, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)
    V3 = ndimage.filters.gaussian_filter1d(gray_video, 8, axis = 0, order= 1, output=None, mode='reflect', cval=0.0, truncate=4.0)

    mean_v1 = np.mean(V1, axis =0)
    mean_v2 = np.mean(V2, axis =0)
    mean_v3 = np.mean(V3, axis =0)
    features_t = [
		np.mean(mean_v1), 
		np.mean(mean_v2), 
		np.mean(mean_v3), 
        np.std(mean_v1),
        np.std(mean_v2),	
        np.std(mean_v3),]
    return features_t


def get_feature_video(video_path):
    color_video, gray_video, _ = read_video(video_path)
    features = []

    features += luminance_feature(color_video)
    features += color_feature(color_video)
    features += rms_contrast_feature(gray_video)
    features += face_num_frame(gray_video)
    features += get_gaussian_spatial(gray_video)
    features += get_gaussian_temporal(gray_video)
    features += SI_feature(gray_video)
    features += TI_frame(gray_video)

    return features


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
    import time

    df = pd.DataFrame(columns=('video','luminance_mean','luminance_std','color_mean','color_std',\
        'contrast_mean','contrast_std','num_face_mean','num_face_std','g_spatial_1_mean','g_spatial_2_mean','g_spatial_3_mean',\
            'g_spatial_4_mean','g_spatial_5_mean','g_spatial_6_mean','g_spatial_1_std','g_spatial_2_std','g_spatial_3_std',\
            'g_spatial_4_std','g_spatial_5_std','g_spatial_6_std','g_temporal_1_mean','g_temporal_2_mean','g_temporal_3_mean',\
                'g_temporal_1_std','g_temporal_2_std','g_temporal_3_std','SI','TI'))

    # total_feature = []

    video_path = '/Users/yuyu/database/TaoLive/Video/'
    for i, video in enumerate(os.listdir(video_path)) :
        total_feature = []
        total_feature.append(video)

        start = time.time()
        total_feature += get_feature_video(os.path.join(video_path,video))
        print(time.time() - start)
        start = time.time()
        print(total_feature)
        df.loc[i] = total_feature

        if i == 2:
            break
    
    # print(df)
    df.to_csv('feature.csv', index= False)




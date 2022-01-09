import numpy as np
from scipy import stats
import pandas as pd

def reject_accept(data_array):
    # input:the subjective data, a numpy array with size:(n_videos, n_persons), dtype:float
    # output: 
    # reject: reject_person, size:(1, n_persons)
    # 1: the person's all subjective data is rejected, 0 means not
    # accept_idx: a numpy array with size:(n_videos, n_persons), 1: a person's data for a video is accepted
    scoreMeanvideo = np.mean(data_array, axis=1)
    scoreStdvideo = np.std(data_array, axis=1)

    reject = np.zeros([1, data_array.shape[1]])
    accept_idx = np.zeros_like(data_array)
    beta = stats.kurtosis(data_array.T, fisher=False)   # the kurtosis for each video

    for person in range(data_array.shape[1]):
        P = 0.0
        Q = 0.0
        for video in range(data_array.shape[0]):
            k = 2 if (beta[video] >= 2 and beta[video] <= 4) else np.sqrt(20)

            if data_array[video, person] >= (scoreMeanvideo[video] + k * scoreStdvideo[video]):
                P += 1
            elif data_array[video, person] <= (scoreMeanvideo[video] - k * scoreStdvideo[video]):
                Q += 1

        # if (P+Q)/data_array.shape[0] > 0.05 and abs((P-Q)/(P+Q)) < 0.3:
        if (P+Q)/data_array.shape[0] > 0.1 and abs((P-Q)/(P+Q)) < 0.3:
            reject[0, person] = 1
    
    accept = 1 - reject
    for person in range(data_array.shape[1]):
        if accept[0, person] == 1:
            for video in range(data_array.shape[0]):
                kk = 2 if (beta[video] >= 2 and beta[video] <= 4) else np.sqrt(20)
                if (data_array[video, person] >= (scoreMeanvideo[video] - kk * scoreStdvideo[video])) and \
                    (data_array[video, person] <= (scoreMeanvideo[video] + kk * scoreStdvideo[video])):
                        accept_idx[video, person] = 1
        else:
            accept_idx[:, person] = 0
    
    return reject, accept_idx

def compute_MOS(data_array, accept_idx):
    MOS = np.zeros([1, data_array.shape[0]])
    for video in range(data_array.shape[0]):
        MOS[0, video] = np.mean(data_array[video, np.where(accept_idx[video, :]==1)])

    return MOS

def compute_MOSz(data_array, accept_idx):
    MOSz = np.zeros([1,data_array.shape[0]])
    scoreMeanPerson = np.mean(data_array, axis=0)
    scoreStdPerson = np.std(data_array, axis=0)
    data_arrayz= np.zeros_like(data_array)
    for person in range(data_arrayz.shape[1]): 
        data_arrayz[:, person] = (data_array[:, person] - scoreMeanPerson[person])/scoreStdPerson[person]

    data_arrayz = (data_arrayz+3)*100/6
    for video in range(data_arrayz.shape[0]):
        MOSz[0, video] = np.mean(data_arrayz[video, np.where(accept_idx[video, :]==1)])

    return MOSz

def data_cleaning(data_array):
    # input:the subjective data, a numpy array with size:(n_videos, n_persons), dtype:float
    # output: a numpy array with size:(n,)
    # reject: reject_person
    # 1: the person's all subjective data is rejected, 0 means not
    # person_outliers, video_outliers: the number of outliers for each person/video
    # MOS, MOSz
    reject, accept_idx = reject_accept(data_array)
    person_outliers = np.sum(1 - accept_idx, axis=0)
    video_outliers = np.sum(1 - accept_idx, axis=1)

    MOS = compute_MOSz(data_array, accept_idx)
    # MOSz = compute_MOSz(data_array, accept_idx)

    # return reject[0], person_outliers, video_outliers, MOS[0], MOSz[0]
    return reject[0], person_outliers, video_outliers, MOS[0]
    

if __name__ == '__main__':

    # load the data
    # data = pd.read_csv('{}测试结果.csv'.format(term), index_col=None, header=None)
    data = pd.read_csv('MOS_var_test_no_filter.csv')
    data = np.array(data.iloc[:,2:-2])
    # data = np.random.randint(101, size=(22, 20))
    data_float = np.array(data).astype(float)
    # data_float = data_float.T
    print ('(num_videos, num_persons):', data_float.shape)

    # reject, person_outliers, video_outliers, MOS, MOSz = data_cleaning(data_float)
    reject, person_outliers, video_outliers, MOS = data_cleaning(data_float)
    print ('The list of rejection:', reject)
    print ('The number of outliers for each person:', person_outliers)
    print ('The number of outliers for each video:', video_outliers)
    # print ('The MOS for each video:', MOS)
    # print ('The MOSz for each video:', MOSz)
    df_ref = pd.read_csv('CG_QA_test.csv')

    # data_mos = {'MOS':MOS, 'MOSz': MOSz}
    data_mos = {'Image':df_ref['Image'],'MOS':np.round(MOS,2)}
    df = pd.DataFrame(data_mos)
    df.to_csv('MOS_var_test_norm_filter.csv', index=0)

    # df2 = pd.DataFrame({'outliers':person_outliers.tolist()})  
    # df2.to_csv('subject_outliers.csv')

    # df3 = pd.DataFrame({'outliers':video_outliers.tolist()})
    # df3.to_csv('video_outliers.csv')

#     The list of rejection: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# The number of outliers for each person: [ 57.  47.  79.  30.  34. 137.  55. 103.  36.  28.  42.  70.  13.   7.
#    7.   9.   9.]
# The number of outliers for each video: [1. 0. 1. ... 1. 0. 0.]


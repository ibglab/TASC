from scipy.signal import savgol_filter
from scipy import interpolate
from scipy import signal as sig
from scipy.signal import butter, filtfilt

import numpy as np
import random
import os
import pickle

from numba import jit

from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances

import logging
logging.basicConfig(

    level=logging.INFO, format="%(levelname)s: %(asctime)s: %(message)s"
)

def runs_of_ones(arr):
    # find runs of ones
    diff = np.diff(arr, prepend=0, append=0)
    ones_b = np.where(diff == 1)[0]
    ones_e = np.where(diff == -1)[0] - 1
    return list(zip(ones_b, ones_e))

def runs_of_zeros(arr):
    # find runs of ones
    diff = np.diff(arr, prepend=1, append=1)
    ones_b = np.where(diff == -1)[0]
    ones_e = np.where(diff == 1)[0] - 1
    return list(zip(ones_b, ones_e))

# according to config, set the dimentions of the input data (landmarks or distances)
def get_vame_input_type(project_path, file_signature):
    curr_path = os.path.join(project_path, 'data')
    temp = [ele for ele in os.listdir(curr_path) if 'train' not in ele][0]
    curr_path = os.path.join(curr_path, temp)
    temp = [ele for ele in os.listdir(curr_path) if file_signature in ele][0]
    temp = np.load(os.path.join(curr_path, temp))
    if temp.shape[0] == 32:
        return 'landmarks'
    else:
        return 'distances'

def find_quiet_periods(lm, sfreq=1, thresh=3, fs=1, min_length=40, bord=10):
    '''smoothes the time series and takes derivative
    calculates time points where accumulated movemenet
    is below the threshold value (indicating the rest)
    
    sfreq - frequency for low pass filter, i.e. only returns signal below a certain frequency (to filter out noise)
    thresh - quiter_frames = signal < thresh. The lower the threshold, the less there are quiet frames
    fs - ???
    min_length - min_length of a segment to be accepted as a quiet period
    bord - how much of an offset to take before cutting the quiet period
    '''
    quiet_periods = []    
    pr_lm = lm.copy()
    b, a = butter(4, sfreq/fs*2, btype='lowpass', fs=fs)
    for i, ts in enumerate(lm.T):
        ts_filt = filtfilt(b, a, ts)
        pr_lm[:,i] = np.diff(ts_filt, prepend=ts[0])
    movement = np.sum(np.abs(pr_lm), axis=1)
    quiet_frames = (movement < thresh) * 1
    qs = np.where(np.diff(quiet_frames) == 1)[0]
    qe = np.where(np.diff(quiet_frames) == -1)[0]
    if len(qs) != 0:
        if qe[0]<qs[0]:
            qs = np.hstack((-bord, qs))
        if len(qs) == len(qe)+1:
            qe = np.hstack((qe, len(quiet_frames)+bord ))
            
        for i in range(len(qs)):
            if qe[i]-qs[i] > min_length:
                quiet_periods.append([qs[i]+bord, qe[i]-bord])
    
    return np.array(quiet_periods, dtype=int), movement

def get_valid_starts(data_length, quiet_periods, time_window, overlap=0.5):
    '''
    outputs an array where
    1 inidicate valid times for movement detection
    0 inidcate likely no movement
    
    overlap - how much of an overlap with rest to consider part of valid starts
    '''
    valid_starts = np.ones(data_length)
    bord = int(time_window * overlap)
    for i, qp in enumerate(quiet_periods): 
        if i == 0:
            valid_starts[max(qp[0]-bord, 0):qp[1]-bord] = 0
        else:
            valid_starts[qp[0]-bord:qp[1]-bord] = 0
    return valid_starts


def load_tacom(path):
    if not os.path.exists(path):
        print(f'{path} does not exist')
        return -1
    with open(path, 'rb') as handle:
        return pickle.load(handle)
    
    
def save_tacom(to_save, path, override=False):
    if os.path.exists(path) and not override:
        print(f'{path} already exists. Pass override=True to write over.')
    else:
        with open(path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'object saved at {path}')


def gen_colors(n_colors):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n_colors)]
    return colors


def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.full([len(arrs), np.max(lens)], np.nan)
    for idx, l in enumerate(arrs):
        arr[idx, :len(l)] = l.flatten()
    
    avg = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return avg, std


def random_with_step(max_idx, size, step):
    res = sorted(np.random.randint(0, max_idx, size))

    to_fix = np.where(np.diff(res) < step)[0]

    while len(to_fix) > 0:
        temp = np.random.randint(max_idx)
        res[to_fix[0]] = temp
        to_fix = np.where(np.diff(sorted(res)) < step)[0]

    return res


# Helper function to return indexes of nans
def nan_helper(y):
    '''
    Helper returning indeces of Nans
    '''
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates all nan values of given array
def interpol(arr):
    '''
    Custom interplation function
    Interpolates only over Nans
    '''

    y = np.transpose(arr)

    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])

    arr = np.transpose(y)

    return arr


def smooth_landmarks(data_mat, confidence=0.5, savgol_window = 5, savgol_order = 1, smooth=False):
    '''
    Use for better results when using the DLC outputs
    Replaces low likelihood points with Nans and interpolates over them

    params:
    data_mat: the landmarks array in the format [[x11, y11, l11, x12, y12, l12, x13, y13, l13], ...]
    where x11,y11,l11 - a joint; x12,y12,l12 - another joint and etc. the first index is time.
    (achieved by pd.DataFrame.to_numpy())
    be careful to pass a copy unless you want and accept changes in the original array

    returns:
    smooth array
    '''

    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3 : (i + 1) * 3])
    
    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan
    # interpolate over low confidence areas
    for i in pose_list:
        i = interpol(i)
        # smooth after interpolation:
        if smooth:
            logging.info('smoothing...')
            i[:,0] = savgol_filter(i[:,0], savgol_window, savgol_order)
            i[:,1] = savgol_filter(i[:,1], savgol_window, savgol_order)

    return pose_list


def my_resample(mul_d_arr, t, o_num, n_num):
    '''
    upsampling method that uses interpolation
    usually leads to less distortions
    '''
    range0 = np.linspace(0, t, int(t*o_num))
    range1 = np.linspace(0, t, int(t*n_num))
    
    y = []
    
    if len(mul_d_arr.shape) == 1:
        f = interpolate.interp1d(range0, mul_d_arr, fill_value="extrapolate")
        y.append(f(range1))
    else:
        for dim in range(mul_d_arr.shape[1]):
            f = interpolate.interp1d(range0, mul_d_arr[:, dim], fill_value="extrapolate")
            y.append(f(range1))

    return np.array(list(zip(*y)))


def weightedL2_pair(arr, w=None):
    res = np.zeros((len(arr), len(arr)))
    for i,ar in enumerate(arr):
        res[i] = weightedL2_all(arr, ar, w)
    return res

def weightedL2_all(arr1, arr, w=None):
    res = np.zeros(len(arr1))
    for i,ar in enumerate(arr1):
        res[i] = weightedL2(ar, arr, w)
    return res

@jit(nopython=True) #boundscheck=True is for debugging only
def weightedL2(arr1,arr2,w=None):
    if arr1.ndim < 2:
        a = np.expand_dims(arr1, 1)
        b = np.expand_dims(arr2, 1)
    else:
        a = arr1
        b = arr2
    nonan1 = ~np.isnan(a[:,0])
    nonan2 = ~np.isnan(b[:,0])
    len1 = int(np.count_nonzero(nonan1))
    len2 = int(np.count_nonzero(nonan2))
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    num_feats = int(a.shape[1])

    a_res = np.full((max_len, num_feats), np.mean(a))
    a_res[:len1] = a[nonan1]
    b_res = np.full((max_len, num_feats), np.mean(b))
    b_res[:len2] = b[nonan2]
    
    if w is None:
        w = np.ones(max_len)
        w[:min_len] += np.hanning(min_len)
        w = w[:, None]

    q = a_res-b_res
    q = np.sqrt(np.nansum((w*q*q)))
    return q


def rank_features(centers):
    rank = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            rank.append(np.argmax(weightedL2_feature(centers[i], centers[j])))
    rank = np.unique(rank, return_counts=True)
    order = np.flip(np.argsort(rank[1]))
    return rank[0][order]

def weightedL2_feature(arr1,arr2,w=None):

    if arr1.ndim < 2:
        a = np.expand_dims(arr1, 1)
        b = np.expand_dims(arr2, 1)
    else:
        a = arr1
        b = arr2
    nonan1 = ~np.isnan(a[:,0])
    nonan2 = ~np.isnan(b[:,0])
    len1 = int(np.count_nonzero(nonan1))
    len2 = int(np.count_nonzero(nonan2))
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    num_feats = int(a.shape[1])

    a_res = np.full((max_len, num_feats), np.mean(a))
    a_res[:len1] = a[nonan1]
    b_res = np.full((max_len, num_feats), np.mean(b))
    b_res[:len2] = b[nonan2]
    
    if w is None:
        w = np.ones(max_len)
        w[:min_len] += np.hanning(min_len)
        w = w[:, None]

    q = a_res-b_res
    q = np.sqrt(np.nansum((w*q*q), axis=0))
    return q
from math import degrees
import numpy as np

import scipy.signal as sig
import scipy.optimize as optimize

import lib.utils as utils

import logging

from functools import lru_cache

logging.basicConfig(

    level=logging.INFO, format="%(levelname)s: %(asctime)s: %(message)s"
)

def linear_alignment(X, Y, alpha=0.5, bnds=((-0.1,0.1), (0.8, 1.2)), 
                    crop=False, method=None, criteria='euclidean'):
    """
    Return alignment parameters for X, aligned (to Y) X
    Works for zero knots ONLY
    """
        
    Y_len = len(Y)
    offset = 0
    coeff = 1
    p0 = np.r_[offset, coeff]

    last_cost = 0
    
    @lru_cache(maxsize=128)
    def calc_penalty(p0,p1):
        penalty = abs(p0)
        angle = np.arctan(penalty)
        penalty = angle
        tan_val = np.abs((p1 - 1)) # / (1 + M1 * M2))
        angle = np.arctan(tan_val)
        penalty += 1.5*angle        
        return penalty

    def func(p):
        Y2, (lp,rp) = warp_linear(X, p, crop=crop)
        return Y2, (lp,rp)

    def err(p):
        nonlocal last_cost
        
        if p[1] <= 0:
            return last_cost

        Y2,_ = func(p)
        
        penalty = calc_penalty(*p)
        if criteria == 'correlation':
            intersection_idx = (0,min(Y_len, len(Y2)))
            corr_cost = sum([np.corrcoef(Y[intersection_idx[0]:intersection_idx[1],i],
                                         Y2[intersection_idx[0]:intersection_idx[1],i])[0,1] 
                             for i in range(Y.shape[1])])
            cost = -corr_cost + alpha * penalty
        else:
            w = None
            cost = utils.weightedL2(Y,Y2,w) + alpha * penalty

        last_cost = cost

        return cost
    
    r = optimize.shgo(err, bounds=bnds, n=16, iters=3)
    calc_penalty.cache_clear()
    
    res = func(r.x)
    return err(r.x), r.x, res

def warp_linear(signal, params, crop=False):

    off = params[0]
    slope = params[1]

    signal_shape = signal.shape

    warped_signal = np.zeros(shape=signal_shape)
    my_range = np.arange(0, signal_shape[0])
    # to crop
    ts = []
    # to smooth if we don't want a sharp crop and to preserve the size
    lp, rp = 0, 0

    for t in my_range:
        # piecewise interpolation
        x = t / (signal_shape[0] - 1)
        z = off + slope * x
        if z < 0:
            warped_signal[t] = signal[0]
            lp += 1
        elif z > 1:
            warped_signal[t] = signal[-1]
            rp += 1
        else:
            _i = z * (signal_shape[0] - 1)
            rem = _i % 1
            i = int(_i)
            temp = (1 - rem) * signal[i] 
            if i < len(signal)-1:
                temp += rem * signal[i + 1]
            warped_signal[t] = temp
            ts.append(t)

    # there was simply no signal left
    if len(ts) == 0:
        # return -1
        # should return a special code and deal with it. 
        # res_singal equals warped with no changes only in case of insane warping/no entries...
        print('Signal was not warped properly: either parameters too extreme, or signal too short')
    else:
        ts = np.array(ts)
        warped_signal = warped_signal[ts]
    if not crop:
        warped_signal = np.pad(warped_signal, ((lp,rp), (0,0)), 
                           mode='median')
    
    return warped_signal, [lp, rp] 
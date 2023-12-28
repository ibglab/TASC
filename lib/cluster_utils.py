from sklearn.metrics.pairwise import euclidean_distances, nan_euclidean_distances, pairwise_distances, cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import rankdata

from itertools import repeat, combinations
# from tqdm import tqdm
from tqdm.notebook import trange, tqdm

from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient, softdtw_barycenter

from intervaltree import Interval, IntervalTree

import numpy as np
import numpy.linalg as linalg
import scipy.optimize as optimize

import lib.linear_utils as lu
import lib.utils as u

from contextlib import closing
import multiprocessing
import collections
import os

import timeit

import signal as osignal

import logging
logging.basicConfig(

    level=logging.INFO, format="%(levelname)s: %(asctime)s: %(message)s"
)
     

# initialize the worker process
def init_worker():
    # get the pid for the current worker process
    print(os.getpid())
    # the children should ignore Ctrl-C
    osignal.signal(osignal.SIGINT, osignal.SIG_IGN)
    

def align_all_sequentially(signal, boundaries, labels, time_off=np.arange(-1, 2), len_off=np.arange(-1, 2), 
                           alpha=0.5, filtparam=0.75,
                           valid_starts=None, reinsert_gaps=True,
                           centermode='median', centerreal=True, centercrop=None, 
                           criteria='euclidean',crop=False):
    '''
    Aligns the space segment by segment
    TOOD:    optional - provide precalculated centers...
    
    Input
    signal            - original data
    boundaries        - array of boundaries of segments, n x 2
    labels            - array of labels of segments, n x 1
    time_off          - array of temporal offsets for variants
    len_off           - array of lengths for variants
    alpha             - parameter for linear alignment
                      (higher alpha - cost consists more of the vectors' similarity,
                      lower - of function's complexity)
    
    Output
    align_boundaries  - aligned boundaries for each variant of each segment
    align_params      - parameters of alignment for each variant of each segment
    align_idx         - indeces of segments' boundaries that are already aligned  
    gap_idx           - indeces of segments' boundaries that are not aligned (gaps)
    filter_idx        - indeces of retained boundaries after the initial filtering
    centers           - temporal centers of clusters
    '''   
    
    print(f'Filtering out clusters with filt_param={filtparam}')
    filter_idx, filter_dists, filter_ranks = filter_clusters(get_segments(signal, boundaries), labels, filtparam)
    f_boundaries = boundaries[filter_idx]
    f_labels = labels[filter_idx]

    # calculate clean cluster centers
    _, min_size, max_size, cluster_sizes = get_sizes(f_boundaries, f_labels)
    
    print_cluster_sizes(cluster_sizes)
    
    centers = get_centers(signal, f_boundaries, f_labels, cluster_sizes, mode=centermode, real=centerreal,
                            crop=centercrop)
    
    if valid_starts is None:
        valid_starts = np.ones(len(signal), dtype=int)
    variants = get_variants(signal, f_boundaries, f_labels, centers, valid_starts, time_off, len_off, alpha, criteria, crop, cluster_sizes)
    
    align_boundaries = {}
    align_costs = {}
    align_params = {}
    corr_i = 0
    total = len(filter_idx)
    n_workers = min(len(variants), 8)
    n_workers = 8
    print(f'Starting {n_workers} processes. PIDs:')
    with multiprocessing.Pool(n_workers, initializer=init_worker) as pool:
        try:
            with tqdm(total=total) as pbar:
                for res in pool.imap_unordered(score_temp_variants_names_unpack, variants, chunksize=4):      
                    i, b, (vb_boundaries, vb_costs, vb_params) = res
                    if len(vb_boundaries) == 0: 
                        filter_idx = np.delete(filter_idx, i-corr_i)
                        corr_i += 1
                    else:
                        vb_boundaries += b
                        align_boundaries[i] = vb_boundaries
                        align_costs[i] = vb_costs
                        align_params[i] = vb_params

                    pbar.update()
        except KeyboardInterrupt:
            print('Aborted')
            return 
    
    align_boundaries = list(collections.OrderedDict(sorted(align_boundaries.items())).values())
    align_costs = list(collections.OrderedDict(sorted(align_costs.items())).values())
    align_params = list(collections.OrderedDict(sorted(align_params.items())).values())

    print('Finished multiprocessed alignment')
    print('Starting post-processing')

    align_boundaries = np.concatenate(align_boundaries)
    align_params = np.concatenate(align_params, dtype=object)
    scaler = MinMaxScaler()
    align_costs = np.concatenate([scaler.fit_transform(np.expand_dims(ele,1)).flatten() for ele in align_costs], dtype=object)

    boundaries_idx = align_boundaries[:,0].argsort()
    align_boundaries = align_boundaries[boundaries_idx]
    align_costs = align_costs[boundaries_idx]
    align_params = align_params[boundaries_idx]
    
    no_overlap_idx = divide_and_fix(1-align_costs, align_boundaries)
    align_boundaries = align_boundaries[no_overlap_idx]
    align_params = align_params[no_overlap_idx]
    
    if reinsert_gaps:
        # fix gaps that look like they could be in a cluster
        align_boundaries, gap_params, gap_idx, align_idx = fix_gaps(signal, align_boundaries, centers, 
                                                        min(np.diff(align_boundaries)),
                                                    max_len=len(signal), valid_starts=valid_starts,
                                                    filter_dists=filter_dists)
        for i,p in zip(gap_idx, gap_params):
            align_params = np.insert(align_params, i, p, axis=0)
    else:
        align_idx = np.arange(len(align_boundaries))
        gap_idx = []
   
    return align_boundaries, align_idx, gap_idx, align_params, filter_idx


def get_variants(signal, boundaries, labels, centers, valid_starts, time_off, len_off, alpha, criteria, crop, cluster_sizes):
    variants = []
    for i,b in enumerate(boundaries):
        # original length
        b_len = b[1]-b[0]                                
        # left possible boundary is min of original/plus negative shift
        idx1 = min(b[0], b[0] + time_off[0])
        while idx1 < 0:
            idx1 += 1
        # right possible boundary is the postivie shift plus b_len plus biggest len offset
        idx2 = max(b[0],b[0] + time_off[-1]) + max(b_len,b_len + len_off[-1])
        while idx2 > len(signal):
            idx2 -= 1
        variants.append((i, idx1, signal[idx1:idx2], b-idx1, centers[labels[i]], time_off, len_off, alpha,
                         valid_starts[idx1:idx2],criteria, crop, 
                         max(2,np.floor(cluster_sizes[labels[i]]*0.8)), np.ceil(cluster_sizes[labels[i]]*1.3)))
    return variants


def score_temp_variants_names_unpack(args):
    return args[0], args[1], score_temp_variants(*args[2:])


def score_temp_variants(signal, boundary, cluster_center, time_off=np.arange(-1,2), len_off=np.arange(-1,2), alpha=0.5, valid_starts=None, criteria='euclidean', crop=False, min_win=0, max_win=999):
    '''
    Extracts and scores variants of a segment
    
    Input
    signal         - temporal data
    boundary       - temporal boundary [b1,b2] of the segment
    cluster_center - the cluster center to score against
    time_off       - array of temporal offsets for variants
    len_off        - array of lengths for variants
    alpha          - parameter for linear alignment 
                    
    Output
    boundaries     - aligned boundaries, num_variants x 2
    costs          - costs of alignments, num_variants x 1
    params         - parameters of alignment, num_variants x num_parameters
    '''      
    if valid_starts is None:
        valid_starts = np.ones(signal.shape[0])
    
    time_len = np.array(np.meshgrid(time_off, len_off)).T.reshape(-1,2)
    og_len = boundary[1] - boundary[0]
    boundaries = np.zeros((len(time_len), 2), dtype=int)
    costs = np.zeros(len(time_len))
    params = np.zeros((len(time_len), 2))
    
    i = 0
    for tl in time_len:
        b1 = boundary[0]+tl[0]
        b2 = b1 + og_len + tl[1]
        if (b1 < 0):
            continue
        if (b2 > len(signal)):
            continue
        if (b2-b1 < min_win) or (b2-b1 > max_win):  
            continue
        if (-1 in valid_starts[b1:b2]) or (sum(valid_starts[b1:b2]) < 0.95*(b2-b1)):
            continue
            
        seg = signal[b1 : b2]

        cost, param, _ = lu.linear_alignment(seg, cluster_center, alpha=alpha, criteria=criteria, crop=crop)

        boundaries[i] = [b1,b2]
        costs[i] = cost
        params[i] = param
        i += 1
        
    boundaries = boundaries[:i]
    costs = costs[:i]
    params = params[:i]
    
    sort_idx = np.argsort(costs)
    costs = costs[sort_idx]
    boundaries = boundaries[sort_idx]
    params = params[sort_idx]
    
    return boundaries, costs, params


def fix_overlaps(costs, boundaries):
    '''
    Returns optimal variant/boundary, such that there are no overlaps while minimizing sum of the costs
    Uses dynamic programming
    Assumes sorted boundaries

    Drawback: assumes there is at least one non-overlapping variant
            for each segment. As a result might return an overlap.
    
    Input
    costs       - cost of alignement for each variant of each segment
    boundaries  - aligned boundaries for each variant of each segment
    
    Output
    path        - array of indeces of the best variant for each segment (sequentially), n x 1
    '''
    T = []
    indeces = []
    
    for i in range(len(costs)):
        t = []
        for j in range(costs[i].shape[0]):
            t = np.r_[t, costs[i][j]]
        T.append(np.array(t)) 
    T = np.array(T, dtype=object)
    
    for i in range(1,len(costs)):
        index = []
        for j in range(0, T[i].shape[0]):
            # 9999 - vague big number
            # this assumes boundaries are sorted!
            prev_costs = np.array([t if not check_overlap(boundaries[i-1][k], boundaries[i][j]) 
                                    else 9999 for k,t in enumerate(T[i-1]) if t is not None])
            if len(np.where(prev_costs==9999)[0]) == len(prev_costs):
                T[i][j] = 9999
                # index = np.r_[index, -1]
            else:
                T[i][j] += T[i-1][np.argmin(prev_costs)]
            index = np.r_[index, np.argmin(prev_costs)]
        indeces.append(index) 

    path = np.zeros((len(costs)), dtype=int)
    path[-1] = np.array([np.argmin(T[-1])])
    for i in range(len(costs)-2, -1, -1):
        path[i] = indeces[i][path[i+1]]
    return path


def fix_singular_overlaps(scores, boundaries):
    '''
    Returns optimal boundaries, such that there are no overlaps while MAXIMIZING
    sum of the scrores
    Uses dynamic programming

    Assumes sorting

    Input
    scores      - score for each variant
    boundaries  - aligned boundaries for each variant of each segment
    
    Output
    path        - array of indeces of the best variant for each segment (sequentially), n x 1
    '''
    
    T = np.full((boundaries.shape[0]+1, boundaries.shape[0]), -9999, dtype=float)
    T[0] = scores
    indeces = {}
    for row in range(1 ,T.shape[0]):
        for col in range(T.shape[1]):
            if boundaries[row-1][0] >= boundaries[col][1]:
                if max(T[col+1]) > T[0,col]:
                    index = np.r_[indeces[(col, np.argmax(T[col+1]))]]                    
                else:
                    index = np.r_[indeces[(col,0)]]

                T[row,col] = T[0,row-1] + max(max(T[col+1]), T[0,col])
                
                if (row-1, col) in indeces.keys():
                    indeces[(row-1, col)].append(index)
                else:
                    indeces[(row-1, col)] = index

            if (row-1, col) in indeces:
                indeces[(row-1, col)] = np.r_[indeces[(row-1, col)], [row-1]]
            else:
                indeces[(row-1, col)] = [row-1]
                
    best_path = np.unravel_index(np.argmax(T), T.shape)
    if best_path[0] > 0:
        best_path = indeces[(best_path[0]-1, best_path[1])]
    else:
        best_path = [best_path[1]]
    
    return best_path


def fix_singular_overlaps_names_unpack(args):
    return args[0], fix_singular_overlaps(*args[1:])


def divide_and_fix(scores, boundaries, max_len=500):
    '''
    Prefered to call that for inputs larger than 500
    Divides input into chunks
    Provides a rougher evaluation due to chunking
    However, is more efficient

    Works on *maximizing* scores
    Not minimizing costs
    '''
    filter_idx = {}
    variants = [(idx, scores[idx:idx+max_len], boundaries[idx:idx+max_len]) for idx in np.arange(0, len(boundaries), max_len)]
    n_workers = min(len(variants), 4)
    with closing(multiprocessing.Pool(n_workers, initializer=init_worker)) as pool:
        with tqdm(total=len(variants)) as pbar:
            for res in pool.imap_unordered(fix_singular_overlaps_names_unpack, variants):
                filter_idx[res[0]] = res[1]+res[0]
                pbar.update()
        pool.terminate()
        
    filter_idx = list(collections.OrderedDict(sorted(filter_idx.items())).values())
    filter_idx = np.concatenate(filter_idx)
        
    res_boundaries = boundaries[filter_idx]
    res_scores = scores[filter_idx]
    cat_overlaps = check_overlaps(res_boundaries)
    to_remove = []
    if len(cat_overlaps) > 0:
        for cp in cat_overlaps:
            if res_scores[cp[0]] > res_scores[cp[1]]:
                to_remove.append(cp[1])
            else:
                to_remove.append(cp[0])

    res_filter_idx = np.array([ele for i, ele in enumerate(filter_idx) if i not in to_remove])

    return res_filter_idx


def fix_gaps(signal, boundaries, centers, win_size, filter_dists=None, max_len=None, valid_starts=None, alpha=2):
    '''
    Re-inserts gaps (either occuring from filtering or from overlap cleanup)
    '''    
    if max_len is None:
        max_len = len(signal)
    if valid_starts is None:
        valid_starts = np.ones(signal.shape[0])

    centers = np.array([item for item in centers.values()])
    
    # Find candidate gaps for insertion (gaps that are >= than the min window size)
    cand_idx = np.where(np.array([boundaries[i+1][0] - boundaries[i][1] for i in range(len(boundaries)-1)]) >= win_size)[0]+1
    gaps = np.r_[[[boundaries[g-1][1], boundaries[g][0]] for g in cand_idx]]
    if max_len-boundaries[-1][1] >= win_size:
        gaps = np.r_[gaps, [[boundaries[-1][1], max_len]]]
    
    res_boundaries = boundaries.copy()
    costs = []
    gap_boundaries = []
    for i,gap in enumerate(tqdm(gaps)):
        t0, t1 = gap
        cands = np.arange(t0, t1-win_size, win_size//3)
        for e,cand in enumerate(cands):
            insert = False
            idx1, idx2 = int(cand), int(cand+win_size)
            if -1 in valid_starts[idx1:idx2]: continue
            if sum(valid_starts[idx1:idx2]) >= 0.95*(idx2-idx1):
                curr_seg = signal[idx1:idx2]
                c = u.weightedL2_all(centers, curr_seg)
                if filter_dists is None:
                    insert = True
                else:
                    clust = filter_dists[np.argmin(c)]
                    insert = (min(c) - clust[0]) < alpha*clust[1]
                if insert:
                    costs.append([[np.argmin(c), min(c)]])
                    gap_boundaries.append([[idx1, idx2]])
    
    gaps_idx = []
    gaps_params = []
    if len(gap_boundaries) == 0:
        print('No gaps filled')
    else:
        costs = np.concatenate(costs)
        scaler = MinMaxScaler()
        for c in np.unique(costs[:,0]):
            c_ind = np.where(costs[:,0]==c)[0]
            costs[c_ind,1] = scaler.fit_transform(np.expand_dims(costs[c_ind, 1],1)).flatten()
        gap_boundaries = np.concatenate(gap_boundaries)
        gap_sort = gap_boundaries[:,0].argsort()
        gap_boundaries = gap_boundaries[gap_sort]
        costs = costs[gap_sort]
        fix_idx = divide_and_fix(1-costs[:,1], gap_boundaries, max_len=500)
        gap_boundaries = gap_boundaries[fix_idx]
        costs = costs[fix_idx,0].astype(int)
        idx = np.searchsorted(res_boundaries[:,0], gap_boundaries[:,0])
        res_boundaries = np.insert(res_boundaries, idx, gap_boundaries, axis=0)
        gaps_idx = idx + np.arange(len(idx))
        print(f'Inserted {len(gaps_idx)} gaps')
        for i,gp in enumerate(gap_boundaries):
            gaps_params.append(lu.linear_alignment(signal[gp[0]:gp[1]], centers[costs[i]], crop=False)[1])
    
    og_idx = [i for i in np.arange(res_boundaries.shape[0]) if i not in gaps_idx]

    return res_boundaries.astype(int), np.array(gaps_params), np.array(gaps_idx, dtype=int), np.array(og_idx, dtype=int)


def get_sizes(boundaries, labels):
    """
    Calculates median length of a segment for each cluster
    Return median over the clusters, min, max and a dict per cluster
    """
    ulabels = np.unique(labels)
    clusters_sizes = np.zeros(len(ulabels))
    cluster_sizes_dict = {}
    for e, lab in enumerate(ulabels):
        c_ind = np.where(labels==lab)[0]
        clusters_sizes[e] = np.median(boundaries[c_ind][:,1] - boundaries[c_ind][:,0])
        cluster_sizes_dict[lab] = clusters_sizes[e].astype(int)
    return np.median(clusters_sizes), np.min(clusters_sizes), np.max(clusters_sizes), cluster_sizes_dict


def inter_cluster_distances(labels, distances, method='nearest'):
    """
    Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """    
    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn_index(X, labels, diameter_method='farthest', cdist_method='nearest'):
    '''
    Operates in latent domain
    '''
    distances = pairwise_distances(X)
    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    diameters = diameter(labels, distances, diameter_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameters)

    return min_distance / max_diameter


def avg_dists(X, labels, n_clusters):        
        
    LABELS = np.unique(labels)

    centers = {}
    for l in LABELS:
        c_ind = np.where(labels==l)[0]
        centers[l] = get_center(X[c_ind], dist='cosine')[1]
    
    average_inner_dist = np.zeros(n_clusters)
    average_cluster_size = []
    for l in LABELS:
        c_ind = np.where(labels==l)[0]
        average_cluster_size.append(len(c_ind))
        cluster_vectors = X[c_ind, :]
        # average distance of cluster members to their center
        # calculated on latent space
        average_inner_dist_temp = (1-cosine_similarity(cluster_vectors)[np.triu_indices(len(c_ind), 1)])
        average_inner_dist[l] = average_inner_dist_temp.mean()
    
    # distances from center to center
    # in latent space
    outter_dists = (1-cosine_similarity(list(centers.values())))
    tri_dists = outter_dists[np.triu_indices(n_clusters, 1)]
    max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()

    return average_inner_dist.mean(), avg_dist, average_cluster_size


def eval_clusters(X, labels, metrics=['silhouette', 'cal_har', 'dav_boul', 'dunn', 'avg_dists']):
    evals = {}
    labels = LabelEncoder().fit(labels).transform(labels)

    for metric in metrics:
        # how similar members are to their cluster compares to the others
        if metric == 'silhouette':
            evals[metric] = silhouette_score(X, labels)
        # ratio of sum of between-cluster dispersion / within-cluster dispersoin
        if metric == 'cal_har':
            evals[metric] = calinski_harabasz_score(X, labels)
        # cluster similarity to its closest cluster
        # ratio within-cluster distances / between-cluster distances
        if metric == 'dav_boul':
            evals[metric] = davies_bouldin_score(X, labels)
        if metric == 'dunn':
            evals[metric] = dunn_index(X, labels)
        if metric == 'avg_dists':
            temp = avg_dists(X, labels, len(np.unique(labels)))
            evals['avg_in'] = temp[0]
            evals['avg_out'] = temp[1]

    return evals


def get_segments(signal, boundaries, nan=True):
    '''
    Returns segments of the signal based on the boundaries
    '''
    if not nan:
        res = []
        for b in boundaries:
            res.append(signal[b[0]:b[1]])
        try:
            res = np.array(res, dtype=float)
        except:
            res = np.array(res, dtype=object)
    else:
        max_len = max([b[1]-b[0] for b in boundaries])
        res = np.full((boundaries.shape[0], max_len, signal.shape[1]), np.nan, dtype=float)
        for i,b in enumerate(boundaries):
            res[i, :b[1]-b[0], :] = signal[b[0]:b[1]]
    return res


def filter_clusters(segments, labels, alpha=0.75, dist='weight'):
    '''
    segments     - segments in laent or temporal domain
    labels       - labels for the segments
    alpha        - number of std from the mean to preserve
    dist         - default: 'weight' - calculates weighted L2 (temporal domain)
                   pass 'cosine' for the embedded space
    
    returns
    filter_idx   - indeces of preserved members, concatenated
    filter_dists - statistics per cluster, median and std of the distances
    ranks        - ranks for each cluster (preserving order), 
                   low value means lower distance (better member)
    '''
    ulabels = np.unique(labels)
    filter_idx = []
    filter_dists = np.zeros((len(ulabels), 2))
    ranks = {}
    offset = 0
    for i,lab in enumerate(tqdm(ulabels, desc='Filtering clusters')):
        cluster = segments[np.where(labels==lab)[0]]
        f_idx, f_dist, f_rank = get_filtered_cluster(cluster, alpha=alpha, dist=dist)
        filter_idx = np.r_[filter_idx, offset+f_idx]
        filter_dists[i] = f_dist
        offset += len(cluster)
        ranks[lab] = f_rank
    filter_idx = filter_idx.astype(int)
    return filter_idx, filter_dists, ranks


def get_filtered_cluster(cluster, alpha=0.75, dist='weight'):
    '''
    Return index of the 'best' cluster members
    Based on the weighted pairwise distances of the members
    or on cosine similarity for latent space
    cluster       - members of a cluster
    alpha         - number of std from the mean to preserve
    dist          - the type of distance to use (weight L2 or cosine)
    
    returns
    cluster_idx   - indeces of preserved members
    [median, std] - statistics per cluster, mean and std of the distances
    rank          - ranks (preserving order), 
                    low value means lower distance (better member)
    TODO: clcualte sum of distances instead?
    '''
    # pairwise distance
    if dist == 'weight':
        dists = u.weightedL2_pair(cluster)
    else:
        dists = 1-cosine_similarity(cluster)
    
    mem_medians = np.median(dists, axis=0)
    m_median = np.median(mem_medians)
    m_std = np.std(mem_medians)
    
    cluster_idx = np.where((mem_medians-m_median) < alpha*m_std)[0]
    
    # returns lower rank for lower distance
    rank = rankdata(mem_medians)
    
    return np.array(cluster_idx), [m_median, m_std], rank.astype(int)


def get_center(cluster_vectors, mode='median', cluster_size=-1, dist='weight'):
    '''
    Returns the cluster center and the actual cluster memeber closest to the center
    <centeral_member>, <central_value>
    If mode='median' - returns median
    If more='mean' - returns mean
    
    '''
    if cluster_size != -1:
        cluster_vectors = cluster_vectors[:,:cluster_size]

    try:
        # or np.median
        if mode=='median':
            # print('Calculating cluster median')
            center_value = np.nanmedian(cluster_vectors, axis=0)
        elif mode=='mean':
            # print('Calculating cluster mean')
            center_value = np.nanmean(cluster_vectors, axis=0)
        elif mode=='euc_barycenter':
            print('Calculating euclidean barycenter')
            center_value = euclidean_barycenter(cluster_vectors)
        elif mode=='dtw_barycenter':
            print('Calculating dtw barycenter')
            center_value = dtw_barycenter_averaging(cluster_vectors)
        elif mode=='softdtw_barycenter':
            print('Calculating soft dtw barycenter')
            center_value = softdtw_barycenter(cluster_vectors)
    except Exception as e:
        print(e)
        center_value = np.nanmean(cluster_vectors, axis=0)
    
    if dist == 'weight':
        dist = u.weightedL2_all(cluster_vectors, center_value)
    else:
        dist = 1-cosine_similarity(cluster_vectors, [center_value]).flatten()
    center_id = np.argmin(dist)
    
    return center_id, center_value


def get_centers(signal, boundaries, labels, cluster_sizes=None, mode='median', real=False, crop=None):
    '''
    get centers of clusters
    
    Inputs
    data    - array, n x dim (x dim)
    labels  - array, nx1
    '''
    u_labels = np.unique(labels)
    res = {}
    data = get_segments(signal, boundaries)
    if cluster_sizes is None:
        cluster_sizes = np.full(len(u_labels), -1)
    for e,l in enumerate(u_labels):
        c_ind = np.where(labels==l)[0]
        center = get_center(data[c_ind], mode, cluster_sizes[l])
        if real:
            b = boundaries[c_ind][center[0]].copy()
            if crop is not None:
                b[0] = max(0,b[0]+crop[0])
                b[1] = min(len(signal),b[1]+crop[1])
            temp = signal[b[0]:b[1]]
            res[l] = temp
        else:
            res[l] = center[1]
            nan_res = u.nan_helper(res[l])
            if len(nan_res[0].shape) > 1:
                res[l] = res[l][~nan_res[0][:,0],:]
            else:
                res[l] = res[l][~nan_res[0]]
            if crop is not None:
                res[l] = res[l][max(0,crop[0]):min(len(res[l]),crop[1])]
    return res

def check_overlaps(arr):
    intervals = []
    for i, (start, end) in enumerate(arr):
        intervals.append(Interval(start, end, i))

    interval_tree = IntervalTree(intervals)

    overlaps = []
    for interval in interval_tree:
        overlapping_intervals = interval_tree.overlap(interval)
        for overlap in overlapping_intervals:
            if overlap != interval and overlap.data > interval.data:  # Avoid self-overlaps
                overlaps.append((interval.data, overlap.data))
    overlaps = np.array(overlaps)
    if len(overlaps) > 0:
        overlaps = overlaps[overlaps[:,0].argsort()]
    return overlaps


def check_overlap(ele1, ele2):
    '''
    Returns true if two segments overlap
    Input: temporal boundaries of the segments
    '''
    range1 = np.arange(ele1[0], ele1[1])
    range2 = np.arange(ele2[0], ele2[1])
    
    return len(set(range1).intersection(range2)) != 0


def full_nans(array):
    '''
    Returns uniformly shaped numpy array
    from a collection of lists of different lengths
    Fills empty entries with nans
    '''
    max_len = max([len(a) for a in array])
    num_dim = array[0].shape[1]
    res = np.full((len(array), max_len, num_dim), np.nan)
    for i,a in enumerate(array):
        res[i, :len(a)] = a
    return res


def flat_concat_segments(segments):
    res = np.zeros((segments.shape[0], segments.shape[1]*segments.shape[2]))
    for i,seg in enumerate(segments):
        res[i] = seg.T.flatten()
    return res


def check_key_add(dictionary, key):
    '''
    Utility function
    Returns a 'safe' key for the given dictionary not to override any existing entries
    Key in format '<name>{version}'
    '''
    keys = list(dictionary.keys())
    while key in keys:
        key = str(int(keys[-1])+1)
    return key


def estimate_fuzzifier(N_data_points: int, M_feat_dim: int):
    """Estimate the fuzzifier for fuzzy c-means using the empirical formula of 
    Schwämmle and Jensen. Bioinformatics (2010)
    Paper at: https://academic.oup.com/bioinformatics/article/26/22/2841/227572
    Katharina's code
    Args:
        N_data_points (int): number of datapoints
        M_feat_dim (int): feature dimension
    """

    return (
        1
        + (1418 / N_data_points + 22.05) * M_feat_dim ** -2
        + (12.33 / N_data_points + 0.243)
        * M_feat_dim ** (-0.0406 * np.log(N_data_points) - 0.1134)
    )


def divide_by_session(data_bounds, sess_bounds, data_to_divide=None):
    '''
    Divides data into sessions
    '''
    bounds_flag = False
    if data_to_divide is None:
        data_to_divide = data_bounds
        bounds_flag = True

    res = {sess: [] for sess in sess_bounds.keys()}
    bound_iter = iter(sess_bounds.items() )
    key, bounds = next(bound_iter)
    prev_sess_shape = 0

    for i,sample in enumerate(data_bounds):
        if (sample[0] >= bounds[0]) and (sample[1] < bounds[1]):
            if bounds_flag:
                res[key].append(data_to_divide[i] - prev_sess_shape)
            else:
                res[key].append(data_to_divide[i])
        else:
            try:
                prev_sess_shape += bounds[1] - bounds[0]
                key, bounds = next(bound_iter)

                if bounds_flag:
                    res[key].append(data_to_divide[i] - prev_sess_shape)
                else:
                    res[key].append(data_to_divide[i])
            except:
                logging.info(f'No more sessions, segments are out of scope, stopped at {i}')
    for key in res.keys():
        res[key] = np.array(res[key])

    return res


def print_cluster_sizes(cluster_sizes):
    sub_clusters = np.random.choice(list(cluster_sizes.keys()), min(len(cluster_sizes), 10), replace=False)
    sub_clusters.sort()
    pstring = str({key:cluster_sizes[key] for key in sub_clusters})
    print(f'Size of a subgroup of clusters: {pstring}')
    
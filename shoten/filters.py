"""Filters meant to reduce noise."""

from math import ceil

import numpy as np


def print_changes(phase, old_len, new_len):
    'Report on absolute and relative changes.'
    coeff = 100 - (new_len/old_len)*100
    print(phase, 'removed', old_len - new_len, '%.1f' % coeff, '%')


def hapax_filter(myvocab):
    '''Eliminate hapax legomena and delete same date only.'''
    old_len = len(myvocab)
    for token in [t for t in myvocab if np.unique(myvocab[t]).shape[0] <= 2]:
        del myvocab[token]
    print_changes('sameness/hapax', old_len, len(myvocab))
    return myvocab


def shortness_filter(myvocab, threshold=7):
    '''Eliminate short words'''
    old_len = len(myvocab)
    for token in [t for t in myvocab if len(t) < threshold]:
        del myvocab[token]
    print_changes('short words', old_len, len(myvocab))
    return myvocab


def frequency_filter(myvocab, max_perc=50, min_perc=.001): # 50 / 0.01
    '''Reduce dict size by stripping least and most frequent items.'''
    old_len = len(myvocab)
    myfreq = np.array([myvocab[l].shape[0] for l in myvocab])
    min_thres, max_thres = np.percentile(myfreq, min_perc), np.percentile(myfreq, max_perc)
    for token in [t for t in myvocab if
                  myvocab[t].shape[0] <= min_thres or myvocab[t].shape[0] > max_thres
                 ]:
        del myvocab[token]
    print_changes('most/less frequent', old_len, len(myvocab))
    return myvocab


def oldest_filter(myvocab, threshold=20):  # 50
    '''Reduce number of candidate by stripping the oldest.'''
    # todo: what about cases like [365, 1, 1, 1] ?
    old_len = len(myvocab)
    myratios = np.array([np.sum(myvocab[l])/myvocab[l].shape[0] for l in myvocab])
    # print(np.percentile(myratios, 40), np.percentile(myratios, 50), np.percentile(myratios, 60))
    threshold = np.percentile(myratios, 50)
    for token in [t for t in myvocab if np.sum(myvocab[t])/myvocab[t].shape[0] > threshold]:
        del myvocab[token]
    print_changes('oldest', old_len, len(myvocab))
    return myvocab


def zipf_filter(myvocab, freqperc=35, lenperc=65):
    '''Filter candidates based on a approximation of Zipf's law.'''
    old_len = len(myvocab)
    freqs = np.array([myvocab[l].shape[0] for l in myvocab])
    freqthreshold = np.percentile(freqs, freqperc)
    lengths = np.array([len(l) for l in myvocab])
    lenthreshold = np.percentile(lengths, lenperc)
    for token in [t for t in myvocab if myvocab[t].shape[0] >= freqthreshold and len(t) <= lenthreshold]:
        # print(myvocab[token].shape[0], len(token), token)
        del myvocab[token]
    print_changes('zipf frequency', old_len, len(myvocab))
    return myvocab


def freshness_filter(myvocab, percentage=10):
    '''Define a freshness threshold to model series of token occurrences in time.'''
    old_len = len(myvocab)
    mysums = np.array([np.sum(myvocab[l]) for l in myvocab])
    datethreshold = np.percentile(mysums, percentage)
    deletions = list()
    for token in myvocab:
        thresh = myvocab[token].shape[0]*(percentage/100)
        freshnessindex = np.sum(myvocab[token][-ceil(thresh):])
        oldnessindex = np.sum(myvocab[token][:ceil(thresh)])
        if oldnessindex < datethreshold:
            #if oldnessindex < np.percentile(myvocab[token], percentage):
            #    continue
            thresh = myvocab[token].shape[0]*(percentage/100)
            freshnessindex = np.sum(myvocab[token][-ceil(thresh):])
            if freshnessindex < np.percentile(myvocab[token], percentage):
                deletions.append(token)
            # print(myvocab[token], freshnessindex, oldnessindex, token)
    for item in deletions:
        del myvocab[item]
    print_changes('freshness', old_len, len(myvocab))
    return myvocab

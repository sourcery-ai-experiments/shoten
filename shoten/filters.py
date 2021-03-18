"""Filters meant to reduce noise."""


from math import ceil

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_NGRAM_VOC = 15000


def print_changes(phase, old_len, new_len):
    'Report on absolute and relative changes.'
    coeff = 100 - (new_len/old_len)*100
    print(phase, 'removed', old_len - new_len, '%.1f' % coeff, '%')


def combined_filters(myvocab, setting):
    '''Apply a combination of filters based on the chosen setting.'''
    if setting == 'loose':
        return freshness_filter(frequency_filter(hapax_filter(myvocab)))
    elif setting == 'normal':
        return freshness_filter(oldest_filter(frequency_filter(hapax_filter(myvocab))))
    elif setting == 'strict':
        return freshness_filter(oldest_filter(
               frequency_filter(shortness_filter(ngram_filter(hapax_filter(myvocab))), min_perc=10)
               ))


def hapax_filter(myvocab):
    '''Eliminate hapax legomena and delete same date only.'''
    old_len = len(myvocab)
    for token in [t for t in myvocab if np.unique(myvocab[t]).shape[0] <= 2]:
        del myvocab[token]
    print_changes('sameness/hapax', old_len, len(myvocab))
    return myvocab


def shortness_filter(myvocab, threshold=20):
    '''Eliminate short words'''
    old_len = len(myvocab)
    lengths = np.array([len(l) for l in myvocab])
    lenthreshold = np.percentile(lengths, threshold)
    for token in [t for t in myvocab if len(t) < lenthreshold]:
        del myvocab[token]
    print_changes('short words', old_len, len(myvocab))
    return myvocab


def frequency_filter(myvocab, max_perc=50, min_perc=.001): # 50 / 0.01
    '''Reduce dict size by stripping least and most frequent items.'''
    old_len = len(myvocab)
    myfreq = np.array([myvocab[l].shape[0] for l in myvocab])
    min_thres, max_thres = np.percentile(myfreq, min_perc), np.percentile(myfreq, max_perc)
    for token in [t for t in myvocab if
                  myvocab[t].shape[0] < min_thres or myvocab[t].shape[0] > max_thres
                 ]:
        del myvocab[token]
    print_changes('most/less frequent', old_len, len(myvocab))
    return myvocab


def hyphenated_filter(myvocab, perc=50, verbose=False): # threshold in percent
    '''Reduce dict size by deleting hyphenated tokens when the parts are frequent.'''
    deletions, old_len = [], len(myvocab)
    myfreqs = np.array([myvocab[l].shape[0] for l in myvocab])
    threshold = np.percentile(myfreqs, perc)
    for word in [w for w in myvocab if '-' in w]:
        mylist = word.split('-')
        firstpart, secondpart = mylist[0], mylist[1]
        if firstpart in myvocab and myvocab[firstpart].shape[0] > threshold or \
           secondpart in myvocab and myvocab[secondpart].shape[0] > threshold:
            deletions.append(word)
    if verbose is True:
        print(sorted(deletions))
    for item in deletions:
        del myvocab[item]
    print_changes('hyphenated', old_len, len(myvocab))
    return myvocab


def oldest_filter(myvocab, threshold=50):
    '''Reduce number of candidate by stripping the oldest.'''
    # todo: what about cases like [365, 1, 1, 1] ?
    old_len = len(myvocab)
    myratios = np.array([np.sum(myvocab[l])/myvocab[l].shape[0] for l in myvocab])
    threshold = np.percentile(myratios, threshold)
    for token in [t for t in myvocab if np.sum(myvocab[t])/myvocab[t].shape[0] > threshold]:
        del myvocab[token]
    print_changes('oldest', old_len, len(myvocab))
    return myvocab


def zipf_filter(myvocab, freqperc=65, lenperc=35, verbose=False):
    '''Filter candidates based on a approximation of Zipf's law.'''
    # todo: count length without hyphen or punctuation: len(l) - l.count('-')
    old_len = len(myvocab)
    freqs = np.array([myvocab[l].shape[0] for l in myvocab])
    freqthreshold = np.percentile(freqs, freqperc)
    lengths = np.array([len(l) for l in myvocab])
    lenthreshold = np.percentile(lengths, lenperc)
    deletions = []
    for token in [t for t in myvocab if myvocab[t].shape[0] >= freqthreshold and len(t) <= lenthreshold]:
        deletions.append(token)
        # if verbose is True:
            # print(token, len(token), myvocab[token].shape[0])
        del myvocab[token]
    if verbose is True:
        print(sorted(deletions))
    print_changes('zipf frequency', old_len, len(myvocab))
    return myvocab


def freshness_filter(myvocab, percentage=10):
    '''Define a freshness threshold to model series of token occurrences in time.'''
    old_len = len(myvocab)
    mysums = np.array([np.sum(myvocab[l]) for l in myvocab])
    datethreshold = np.percentile(mysums, percentage)
    deletions = list()
    for token in myvocab:
        # re-order
        myvocab[token] = -np.sort(-myvocab[token])
        # thresholds
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


def ngram_filter(myvocab, threshold=90, verbose=False):
    '''Find dissimilar tokens based on character n-gram occurrences.'''
    lengths = np.array([len(l) for l in myvocab])
    minlengthreshold = np.percentile(lengths, 1)
    for i in (70, 65, 60, 55, 50, 45, 40):
        maxlengthreshold = np.percentile(lengths, i)
        mytokens = [t for t in myvocab if minlengthreshold <= len(t) <= maxlengthreshold]
        #print(i, len(mytokens))
        if len(mytokens) <= MAX_NGRAM_VOC:
            break
    if len(mytokens) > MAX_NGRAM_VOC:
        print('Vocabulary size too large, skipping n-gram filtering')
        return myvocab
    old_len = len(myvocab)
    # token cosine similarity
    max_exp = 21
    vectorizer = CountVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0)
    firstset = set(compute_deletions(mytokens, vectorizer, threshold))
    vectorizer = TfidfVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0, sublinear_tf=True, binary=True)
    secondset = set(compute_deletions(mytokens, vectorizer, threshold))
    for token in firstset.intersection(secondset):
        del myvocab[token]
    if verbose is True:
        print(sorted(firstset.intersection(secondset)))
    print_changes('ngrams', old_len, len(myvocab))
    return myvocab


def compute_deletions(mytokens, vectorizer, threshold):
    '''Compute deletion list based on n-gram dissimilarities.'''
    count_matrix = vectorizer.fit_transform(mytokens)
    cosine_similarities = cosine_similarity(count_matrix)  # linear_kernel, laplacian_kernel, rbf_kernel, sigmoid_kernel
    myscores = dict()
    for rownum, _ in enumerate(mytokens):
        #myscores[mytokens[rownum]] = np.median(cosine_similarities[:, rownum]) * (np.log(len(mytokens[rownum]))/len(mytokens[rownum]))
        myscores[mytokens[rownum]] = 1 - np.mean(cosine_similarities[:, rownum])
    # print info
    #if verbose is True:
    #    resultsize = 20 # :resultsize*2
    #    for k, v in sorted(myscores.items(), key=lambda item: item[1], reverse=True)[:resultsize]:
    #        print(k, v)
    #    #print()
    #    for k, v in sorted(myscores.items(), key=lambda item: item[1])[:resultsize]:
    #        print(k, v)
    # process
    mylist = np.array([myscores[s] for s in myscores])
    return [s for s in myscores if myscores[s] >= np.percentile(mylist, threshold)]

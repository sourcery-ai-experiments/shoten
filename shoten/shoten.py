"""Main module."""


import _pickle as cpickle
import pickle
import gzip
import re
import string

from collections import defaultdict
# from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from functools import partial
from os import path, walk # cpu_count
from pathlib import Path

import numpy as np

from simplemma import load_data, lemmatize, simple_tokenizer, is_known
from trafilatura.utils import load_html #, sanitize

from .filters import combined_filters


today = datetime.today()
digitsfilter = re.compile(r'[^\W\d\.-]')


def find_files(dirname):
    for thepath, _, files in walk(dirname):
        yield from (path.join(thepath, fname) for fname in files if Path(fname).suffix == '.xml')


def calc_timediff(mydate):
    # compute difference in days
    try:
        thisday = datetime.strptime(mydate, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None
    diff = today - thisday
    return diff.days


def filter_lemmaform(token, lemmadata):
    # apply filter first
    if len(token) < 5 or len(token) > 50 or token.endswith('-'):
        return None
    token = token.rstrip(string.punctuation)
    if len(token) == 0 or token.isnumeric() or not digitsfilter.search(token):
        return None
    # potential new words only
    if is_known(token, lemmadata) is False and \
    sum(map(str.isupper, token)) < 4 and sum(map(str.isdigit, token)) < 4:
        try:
            return lemmatize(token, lemmadata, greedy=True, silent=False)
        except ValueError:
            # and token[:-1] not in myvocab and token[:-1].lower() not in myvocab:
            # token = token.lower()
            return token
    return None


def dehyphen_vocab(vocab):
    deletions = []
    for wordform in [w for w in vocab if '-' in w]:
        splitted = wordform.split('-')
        candidate = ''.join([t.lower() for t in splitted])
        if wordform[0].isupper():
            candidate = candidate.capitalize()
        # fusion occurrence lists and schedule for deletion
        if candidate in vocab:
            #vocab[candidate].extend(vocab[wordform])
            for timediff in vocab[wordform]:
                vocab[candidate] = np.append(vocab[candidate], timediff)
            deletions.append(wordform)
    for word in deletions:
        del vocab[word]
    return vocab


def read_file(filepath, lemmadata, maxdiff=1000):
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # todo: XML-TEI + XML
    # ...
    # XML-TEI: compute difference in days
    timediff = calc_timediff(mytree.xpath('//date')[0].text)
    if timediff is not None and timediff <= maxdiff:
        # process
        for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
            result = filter_lemmaform(token, lemmadata)
            if result is not None:
                # return tuple
                yield result, timediff


def gen_wordlist(mydir, langcodes, maxdiff=1000):
    # init
    myvocab = defaultdict(list)
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    #with ThreadPoolExecutor(max_workers=1) as executor:  # min(cpu_count()*2, 16)
    #    futures = {executor.submit(read_file, f, lemmadata): f for f in find_files(mydir)}
    #    for future in as_completed(futures):
    #        for token, timediff in future.result():
    #            myvocab[token] = np.append(myvocab[token], timediff)
    for filepath in find_files(mydir):
        for token, timediff in read_file(filepath, lemmadata, maxdiff):
            myvocab[token].append(timediff)
    # post-processing
    for item in dehyphen_vocab(myvocab):
        myvocab[item] = np.array(myvocab[item])
    return myvocab


def load_wordlist(myfile, langcodes=None, maxdiff=1000):
    filepath = str(Path(__file__).parent / myfile)
    myvocab = defaultdict(list)
    if langcodes is not None:
        # load language data
        lemmadata = load_data(*langcodes)
    with open(filepath, 'r', encoding='utf-8') as filehandle:
        for line in filehandle:
            columns = line.strip().split('\t')
            if len(columns) != 2:
                print('invalid line:', line)
                continue
            token, date = columns[0], columns[1]
            # compute difference in days
            timediff = calc_timediff(date)
            if timediff is None or timediff > maxdiff:
                continue
            if langcodes is not None:
                result = filter_lemmaform(token, lemmadata)
                if result is not None:
                    myvocab[token].append(timediff)
            else:
                myvocab[token].append(timediff)
    # post-processing
    for item in dehyphen_vocab(myvocab):
        myvocab[item] = np.array(myvocab[item])
    return myvocab


def pickle_wordinfo(mydict, filepath):
    with gzip.open(filepath, 'w') as filehandle:
        cpickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath):
    with gzip.open(filepath) as filehandle:
        return cpickle.load(filehandle)


def apply_filters(myvocab, setting='normal'):
    if setting not in ('loose', 'normal', 'strict'):
        print('invalid setting:', setting)
        setting = 'normal'
    for wordform in sorted(combined_filters(myvocab, setting)):
        print(wordform)


if __name__ == '__main__':
    print('Shoten.')

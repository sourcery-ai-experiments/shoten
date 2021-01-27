"""Main module."""


import gzip
import pickle
import re

from collections import defaultdict
from datetime import datetime
from functools import partial
from os import listdir, path
from pathlib import Path

import numpy as np

from simplemma import load_data, lemmatize, simple_tokenizer # , text_lemmatizer
from trafilatura.utils import load_html, sanitize


today = datetime.today()
digitsfilter = re.compile(r'[^\W\d\.]', re.UNICODE)


def calc_timediff(mydate):
    # compute difference in days
    try:
        thisday = datetime.strptime(mydate, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None
    diff = today - thisday
    return diff.days


def store_lemmaform(token, timediff, myvocab, lemmadata):
    # apply filter first
    if 5 < len(token) < 50 and digitsfilter.search(token):
        try:
            _ = lemmatize(token, lemmadata, greedy=False, silent=False)
        except ValueError:
            # and token[:-1] not in myvocab and token[:-1].lower() not in myvocab:
            # token = token.lower()
            myvocab[token] = np.append(myvocab[token], timediff)
    return myvocab


def gen_wordlist(mydir, langcodes):
    # init
    myvocab = defaultdict(partial(np.array, [], dtype='i'))
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    for filename in listdir(mydir):
        # read data
        with open(path.join(mydir, filename), 'rb') as filehandle:
            mydata = filehandle.read()
        mytree = load_html(mydata)
        # compute difference in days
        timediff = calc_timediff(mytree.xpath('//date')[0].text)
        if timediff is None:
            continue
        # process
        text = sanitize(' '.join(mytree.xpath('//text')[0].itertext()))
        for token in simple_tokenizer(text):
            myvocab = store_lemmaform(token, timediff, myvocab, lemmadata)
    return myvocab


def load_wordlist(myfile, langcodes=None):
    filepath = str(Path(__file__).parent / myfile)
    myvocab = defaultdict(partial(np.array, [], dtype='i'))
    known = set()
    if langcodes is not None:
        # load language data
        lemmadata = load_data(langcodes)
        for dictionary in lemmadata:
            known.update(dictionary.values())
    with open(filepath, 'r', encoding='utf-8') as filehandle:
        for line in filehandle:
            line = line.strip()
            columns = line.split('\t')
            if len(columns) != 2:
                print('invalid line:', line)
                continue
            token, date = columns[0], columns[1]
            # compute difference in days
            timediff = calc_timediff(date)
            if timediff is None:
                continue
            if langcodes is not None:
                # load language data
                lemmadata = load_data(langcodes)
                myvocab = store_lemmaform(token, timediff, myvocab, lemmadata, known)
            else:
                myvocab[token] = np.append(myvocab[token], timediff)
    return myvocab


def pickle_wordinfo(mydict, filename):
    filepath = str(Path(__file__).parent / filename)
    with gzip.open(filepath, 'w') as filehandle:
        pickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filename):
    filepath = str(Path(__file__).parent / filename)
    with gzip.open(filepath) as filehandle:
        return pickle.load(filehandle)


if __name__ == '__main__':
    print('Shoten.')

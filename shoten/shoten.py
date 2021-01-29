"""Main module."""


import gzip
import os
import pickle
import re

from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from simplemma import load_data, lemmatize, simple_tokenizer, is_known # , text_lemmatizer
from trafilatura.utils import load_html, sanitize


today = datetime.today()
digitsfilter = re.compile(r'[^\W\d\.]', re.UNICODE)


def find_files(dirname):
    for thepath, _, files in os.walk(dirname):
        yield from ((thepath, fname) for fname in files if Path(fname).suffix == '.xml')


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
    if 5 < len(token) < 50 and digitsfilter.search(token) and not token.endswith('-'):
        # potential new words only
        if is_known(token, lemmadata) is False:
            try:
                lemma = lemmatize(token, lemmadata, greedy=True, silent=False)
                myvocab[lemma] = np.append(myvocab[lemma], timediff)
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
    for pathname, filename in find_files(mydir):
        # read data
        with open(os.path.join(pathname, filename), 'rb') as filehandle:
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
    if langcodes is not None:
        # load language data
        lemmadata = load_data(*langcodes)
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
                myvocab = store_lemmaform(token, timediff, myvocab, lemmadata)
            else:
                myvocab[token] = np.append(myvocab[token], timediff)
    return myvocab


def pickle_wordinfo(mydict, filepath):
    with gzip.open(filepath, 'w') as filehandle:
        pickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath):
    with gzip.open(filepath) as filehandle:
        return pickle.load(filehandle)


if __name__ == '__main__':
    print('Shoten.')

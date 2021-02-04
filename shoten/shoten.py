"""Main module."""


import gzip
import pickle
import re

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
digitsfilter = re.compile(r'[^\W\d\.]', re.UNICODE)


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
    if 5 < len(token) < 50 and digitsfilter.search(token) and not token.endswith('-'):
        # potential new words only
        if is_known(token, lemmadata) is False and len([l for l in token if l.isupper()]) < 4:
            try:
                return lemmatize(token, lemmadata, greedy=True, silent=False)
            except ValueError:
                # and token[:-1] not in myvocab and token[:-1].lower() not in myvocab:
                # token = token.lower()
                return token
    return None


def read_file(filepath, lemmadata):
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # compute difference in days
    timediff = calc_timediff(mytree.xpath('//date')[0].text)
    if timediff is not None:
        # process
        for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
            result = filter_lemmaform(token, lemmadata)
            if result is not None:
                # return tuple
                yield result, timediff


def gen_wordlist(mydir, langcodes):
    # init
    myvocab = defaultdict(partial(np.array, [], dtype='H')) # I
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    #with ThreadPoolExecutor(max_workers=1) as executor:  # min(cpu_count()*2, 16)
    #    futures = {executor.submit(read_file, f, lemmadata): f for f in find_files(mydir)}
    #    for future in as_completed(futures):
    #        for token, timediff in future.result():
    #            myvocab[token] = np.append(myvocab[token], timediff)
    for filepath in find_files(mydir):
        for token, timediff in read_file(filepath, lemmadata):
            myvocab[token] = np.append(myvocab[token], timediff)
    return myvocab


def load_wordlist(myfile, langcodes=None):
    filepath = str(Path(__file__).parent / myfile)
    myvocab = defaultdict(partial(np.array, [], dtype='H')) # I
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
            if timediff is None:
                continue
            if langcodes is not None:
                result = filter_lemmaform(token, lemmadata)
                if result is not None:
                    myvocab[token] = np.append(myvocab[token], timediff)
            else:
                myvocab[token] = np.append(myvocab[token], timediff)
    return myvocab


def pickle_wordinfo(mydict, filepath):
    with gzip.open(filepath, 'w') as filehandle:
        pickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath):
    with gzip.open(filepath) as filehandle:
        return pickle.load(filehandle)


def apply_filters(myvocab, setting='normal'):
    if setting not in ('loose', 'normal', 'strict'):
        print('invalid setting:', setting)
        setting = 'normal'
    for wordform in sorted(combined_filters(myvocab, setting)):
        print(wordform)


if __name__ == '__main__':
    print('Shoten.')

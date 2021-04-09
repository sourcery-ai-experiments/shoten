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
from htmldate.utils import load_html #, sanitize

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
    if len(token) < 5 or len(token) > 50 or token.endswith('-') or token.startswith('@') or token.startswith('#'):
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


def putinvocab(myvocab, wordform, timediff, source, inheadings=False):
    if wordform not in myvocab:
        myvocab[wordform] = dict()
        myvocab[wordform]['time_series'] = []
        myvocab[wordform]['sources'] = []
        myvocab[wordform]['headings'] = False
    myvocab[wordform]['time_series'].append(timediff)
    if source is not None and len(source) > 0:
        myvocab[wordform]['sources'].append(source)
    if inheadings is True:
        myvocab[wordform]['headings'] = True
    return myvocab


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
            for timediff in vocab[wordform]['time_series']:
                vocab[candidate]['time_series'].append(timediff)
            deletions.append(wordform)
    for word in deletions:
        del vocab[word]
    return vocab


def postprocessing(myvocab):
    myvocab = dehyphen_vocab(myvocab)
    for wordform in myvocab:
        myvocab[wordform]['time_series'] = np.array(myvocab[wordform]['time_series'])
    return myvocab


def read_file(filepath, lemmadata, maxdiff=1000, authorregex=None):
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # todo: XML-TEI + XML
    # ...
    # XML-TEI: filter author
    if authorregex is not None:
        author = mytree.xpath('//author')[0].text
        if authorregex.search(author):
            return
    # XML-TEI: compute difference in days
    timediff = calc_timediff(mytree.xpath('//date')[0].text)
    if timediff is None or timediff >= maxdiff:
        return
    # process
    source = mytree.xpath('//publisher')[0].text
    # headings
    headwords = set()
    for heading in mytree.xpath('//fw'):
        if heading.text_content() is not None:
            # print(heading.text_content())
            for token in simple_tokenizer(heading.text_content()):
                headwords.add(token)
    # process
    for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
        inheadings = False
        if token in headwords:
            inheadings = True
        result = filter_lemmaform(token, lemmadata)
        if result is not None:
            # return tuple
            yield result, timediff, source, inheadings


def gen_wordlist(mydir, langcodes=[], maxdiff=1000, authorregex=None):
    # init
    myvocab = dict()
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    #with ThreadPoolExecutor(max_workers=1) as executor:  # min(cpu_count()*2, 16)
    #    futures = {executor.submit(read_file, f, lemmadata): f for f in find_files(mydir)}
    #    for future in as_completed(futures):
    #        for token, timediff in future.result():
    #            myvocab[token] = np.append(myvocab[token], timediff)
    for filepath in find_files(mydir):
        for token, timediff, source, inheadings in read_file(filepath, lemmadata, maxdiff, authorregex):
            myvocab = putinvocab(myvocab, token, timediff, source, inheadings)
    # post-processing
    return postprocessing(myvocab)


def load_wordlist(myfile, langcodes=[], maxdiff=1000):
    filepath = str(Path(__file__).parent / myfile)
    myvocab = defaultdict(list)
    # load language data
    lemmadata = load_data(*langcodes)
    with open(filepath, 'r', encoding='utf-8') as filehandle:
        for line in filehandle:
            columns = line.strip().split('\t')
            if len(columns) == 2:
                token, date, source = columns[0], columns[1], None
            elif len(columns) == 3:
                token, date, source = columns[0], columns[1], columns[2]
            else:
                print('invalid line:', line)
                continue
            # compute difference in days
            timediff = calc_timediff(date)
            if timediff is None or timediff > maxdiff:
                continue
            if len(langcodes) > 0:
                result = filter_lemmaform(token, lemmadata)
                if result is not None:
                    myvocab = putinvocab(myvocab, token, timediff, source)
            else:
                myvocab = putinvocab(myvocab, token, timediff, source)
    # post-processing
    return postprocessing(myvocab)


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

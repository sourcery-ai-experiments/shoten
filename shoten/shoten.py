"""Main module."""


import csv
import gzip
import pickle

from collections import Counter, defaultdict
# from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from os import path, walk # cpu_count
from pathlib import Path

import numpy as np

from simplemma import load_data, lemmatize, simple_tokenizer, is_known
from htmldate.utils import load_html #, sanitize

import _pickle as cpickle

from .filters import combined_filters, is_relevant_input


today = datetime.today()



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


def filter_lemmaform(token, lemmadata, lemmafilter=True):
    if is_relevant_input(token) is False:
        return None
    # potential new words only
    if lemmafilter is True and is_known(token, lemmadata) is True:
        return None
    # lemmatize
    try:
        return lemmatize(token, lemmadata, greedy=True, silent=False)
    except ValueError:
        # and token[:-1] not in myvocab and token[:-1].lower() not in myvocab:
        # token = token.lower()
        return token


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


def read_file(filepath, lemmadata, maxdiff=1000, authorregex=None, lemmafilter=False):
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # todo: XML-TEI + XML
    # ...
    # XML-TEI: filter author
    if authorregex is not None:
        try:
            author = mytree.xpath('//author')[0].text
            if authorregex.search(author):
                return
        # no author string in the document, log?
        except IndexError:
            pass
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
        result = filter_lemmaform(token, lemmadata, lemmafilter)
        if result is not None:
            # return tuple
            yield result, timediff, source, inheadings


def gen_wordlist(mydir, langcodes=[], maxdiff=1000, authorregex=None, lemmafilter=False):
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
        for token, timediff, source, inheadings in read_file(filepath, lemmadata, maxdiff, authorregex, lemmafilter):
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


def gen_freqlist(mydir, langcodes=[], maxdiff=1000, mindiff=0):
    # init
    myvocab, freqs, oldestday, newestday = dict(), dict(), 0, maxdiff
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    for filepath in find_files(mydir):
        # read data
        with open(filepath, 'rb') as filehandle:
            mytree = load_html(filehandle.read())
            # XML-TEI: compute difference in days
            timediff = calc_timediff(mytree.xpath('//date')[0].text)
            if timediff is None or not mindiff < timediff < maxdiff:
                continue
            if timediff > oldestday:
                oldestday = timediff
            if timediff < newestday:
                newestday = timediff
            # extract
            for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
                if is_relevant_input(token) is True:
                    if token not in myvocab:
                        myvocab[token] = dict()
                        myvocab[token]['time_series'] = []
                    myvocab[token]['time_series'].append(timediff)
    # lemmatize
    if len(langcodes) > 0:
        newvocab = dict()
        for token in myvocab:
            lemma = lemmatize(token, lemmadata, greedy=True, silent=True)
            if is_relevant_input(lemma) is True:
                # register lemma and add frequencies
                if lemma not in newvocab:
                    newvocab[lemma] = dict()
                    newvocab[lemma]['time_series'] = []
                newvocab[lemma]['time_series'] = newvocab[lemma]['time_series'] + myvocab[token]['time_series']
        # post-processing
        myvocab = dehyphen_vocab(newvocab)
        #myvocab = newvocab
    # determine bins
    bins = [i for i in range(oldestday, newestday, -1) if oldestday - i >= 7 and i % 7 == 0]
    if len(bins) == 0:
        print('Not enough days to compute frequencies')
        return freqs
    timeseries = [0] * len(bins)
    #print(oldestday, newestday)
    # remove occurrences that are out of bounds: no complete week
    for item in myvocab:
        myvocab[item]['time_series'] = [d for d in myvocab[item]['time_series'] if not d < bins[-1] and not d > bins[0]]
    # remove hapaxes
    deletions = [w for w in myvocab if len(myvocab[w]['time_series']) <= 1]
    for item in deletions:
        del myvocab[item]
    # frequency computations
    freqsum = sum([len(myvocab[l]['time_series']) for l in myvocab])
    for wordform in myvocab:
        # parts per million
        myvocab[wordform]['total'] = float('{0:.3f}'.format((len(myvocab[wordform]['time_series']) / freqsum)*1000000))
        counter = 0
        freqseries = [0] * len(bins)
        mydays = Counter(myvocab[wordform]['time_series'])
        for day in range(oldestday, newestday, -1):
            if day in mydays:
                counter += mydays[day]
            if day % 7 == 0:
                try:
                    freqseries[bins.index(day)] = counter
                    counter = 0
                except ValueError:
                    pass
        myvocab[wordform]['series_abs'] = freqseries
        # spare memory
        myvocab[wordform]['time_series'] = []
        for i in range(len(bins)):
            timeseries[i] += myvocab[wordform]['series_abs'][i]
    # sum up frequencies
    for wordform in myvocab:
        myvocab[wordform]['series_rel'] = [0] * len(bins)
        for i in range(len(bins)):
            try:
                myvocab[wordform]['series_rel'][i] = (myvocab[wordform]['series_abs'][i] / timeseries[i])*1000000
            except ZeroDivisionError:
                pass
        # take non-zero values and perform calculations
        series = [f for f in myvocab[wordform]['series_rel'] if f != 0]
        myvocab[wordform]['stddev'] = float('{0:.3f}'.format(np.std(series)))
        myvocab[wordform]['mean'] = float('{0:.3f}'.format(np.mean(series)))
        # spare memory
        myvocab[wordform]['series_abs'] = []
    return myvocab


def store_freqlist(freqs, filename):
    with open(filename, 'w') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'total', 'mean', 'stddev', 'relfreqs'])
        for entry in sorted(freqs):
            # only store statistically significant entries
            if freqs[entry]['stddev'] == 0:
                continue
            if freqs[entry]['total'] > 1 or (freqs[entry]['total'] > 0.2 and freqs[entry]['stddev'] < freqs[entry]['mean']/2):
                tsvwriter.writerow([entry, freqs[entry]['total'], freqs[entry]['mean'], freqs[entry]['stddev'], freqs[entry]['series_rel']])


if __name__ == '__main__':
    print('Shoten.')

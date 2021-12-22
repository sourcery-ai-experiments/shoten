"""Main module."""


import csv
import gzip
import pickle

from array import array
from collections import Counter, defaultdict
from datetime import datetime
from os import path, walk # cpu_count
from pathlib import Path

import numpy as np

from courlan import extract_domain
from simplemma import load_data, lemmatize, simple_tokenizer, is_known
from htmldate.utils import load_html #, sanitize

import _pickle as cpickle

from .filters import combined_filters, is_relevant_input


TODAY = datetime.today()


def find_files(dirname):
    "Search a directory for files."
    for thepath, _, files in walk(dirname):
        yield from (path.join(thepath, fname) for fname in files if Path(fname).suffix == '.xml')


def calc_timediff(mydate):
    "Compute the difference in days between today and a date in YYYY-MM-DD format."
    try:
        thisday = datetime.strptime(mydate, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None
    return (TODAY - thisday).days


def filter_lemmaform(token, lemmadata, lemmafilter=True):
    "Determine if the token is to be processed and try to lemmatize it."
    # potential new words only
    if lemmafilter is True and is_known(token, lemmadata) is True:
        return None
    # lemmatize
    try:
        return lemmatize(token, lemmadata, greedy=True, silent=False)
    except ValueError:
        return token


def putinvocab(myvocab, wordform, timediff, source, inheadings=False):
    "Store the word form in the vocabulary or add a new occurrence to it."
    if wordform not in myvocab:
        myvocab[wordform] = dict(time_series=array('H'), sources=Counter(), headings=False)
    myvocab[wordform]['time_series'].append(timediff)
    if source is not None and len(source) > 0:
        myvocab[wordform]['sources'].update([source])
    if inheadings is True:
        myvocab[wordform]['headings'] = True
    return myvocab


def prune_vocab(myvocab, first, second):
    "Append characteristics of wordform to be deleted to an other one."
    if second not in myvocab:
        myvocab[second] = dict(time_series=array('H'), sources=Counter(), headings=False)
    myvocab[second]['time_series'] = myvocab[second]['time_series'] + myvocab[first]['time_series']
    try:
        myvocab[second]['sources'] = sum((myvocab[second]['sources'], myvocab[first]['sources']), Counter())
        if myvocab[first]['headings'] is True:
            myvocab[second]['headings'] = True
    # additional info potentially not present
    except KeyError:
        pass
    return myvocab


def dehyphen_vocab(vocab):
    "Remove hyphens in words if a variant without hyphens exists."
    deletions = []
    for wordform in [w for w in vocab if '-' in w]:
        candidate = ''.join([c.lower() for c in wordform if c != '-'])
        if wordform[0].isupper():
            candidate = candidate.capitalize()
        # fusion occurrence lists and schedule for deletion
        if candidate in vocab:
            vocab = prune_vocab(vocab, wordform, candidate)
            deletions.append(wordform)
    for word in deletions:
        del vocab[word]
    return vocab


def refine_vocab(myvocab, lemmadata, lemmafilter=False, dehyphenation=True):
    """Refine the word list, currently: lemmatize, regroup forms with/without hyphens,
       and convert time series to numpy array."""
    if len(lemmadata) > 0:
        changes, deletions = [], []
        for token in myvocab:
            lemma = filter_lemmaform(token, lemmadata, lemmafilter)
            ##if is_relevant_input(lemma) is True:
            if lemma is None:
                deletions.append(token)
            # register lemma and add frequencies
            elif lemma != token:
                changes.append((token, lemma))
                deletions.append(token)
        for token, lemma in changes:
            myvocab = prune_vocab(myvocab, token, lemma)
        for token in deletions:
            del myvocab[token]
    # dehyphen
    if dehyphenation is True:
        myvocab = dehyphen_vocab(myvocab)
    return myvocab


def convert_to_numpy(myvocab):
    "Convert time series to numpy array."
    for wordform in myvocab:
        myvocab[wordform]['time_series'] = np.array(myvocab[wordform]['time_series'])
    return myvocab


def read_file(filepath, maxdiff=1000, authorregex=None):
    "Extract word forms from a XML TEI file generated by Trafilatura."
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # todo: XML-TEI + XML
    # XML-TEI: compute difference in days
    timediff = calc_timediff(mytree.xpath('//date')[0].text)
    if timediff is None or timediff >= maxdiff:
        return
    # XML-TEI: filter author
    # todo: add authorship flag instead?
    if authorregex is not None:
        try:
            if authorregex.search(mytree.find('.//author').text):
                return
        # no author string in the document, log?
        except AttributeError:
            pass
    # source: extract domain from URL first
    url_elem = mytree.find('.//ptr[@type="URL"]')
    if url_elem is not None and url_elem.get('target') is not None:
        source = extract_domain(url_elem.get('target'))
    # use TEI publisher info
    else:
        source = mytree.find('.//publisher').text
    # headings
    bow = [' '.join(h.itertext()) for h in mytree.xpath('//fw')]
    headwords = {t for t in simple_tokenizer(' '.join(bow)) if is_relevant_input(t)}
    # process
    for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
        # form and regex-based filter
        if is_relevant_input(token) is True:
            # return tuple
            yield token, timediff, source, token in headwords


def gen_wordlist(mydir, langcodes=[], maxdiff=1000, authorregex=None, lemmafilter=False):
    """Generate a list of occurrences (tokens or lemmatas) from an input directory
       containing XML-TEI files."""
    # init
    myvocab = {}
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    for filepath in find_files(mydir):
        for token, timediff, source, inheadings in read_file(filepath, maxdiff, authorregex):
            myvocab = putinvocab(myvocab, token, timediff, source, inheadings)
    # post-processing
    myvocab = refine_vocab(myvocab, lemmadata, lemmafilter)
    return convert_to_numpy(myvocab)


def load_wordlist(myfile, langcodes=[], maxdiff=1000):
    """Load a pre-generated list of occurrences in TSV-format:
       token/lemma + TAB + date in YYYY-MM-DD format + TAB + source (optional)."""
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
            # skipping this: if is_relevant_input(token) is True
            myvocab = putinvocab(myvocab, token, timediff, source)
    # post-processing
    myvocab = refine_vocab(myvocab, lemmadata)
    return convert_to_numpy(myvocab)


def pickle_wordinfo(mydict, filepath):
    "Store the frequency dict in a compressed format."
    with gzip.open(filepath, 'w') as filehandle:
        cpickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath):
    "Open the compressed pickle file and load the frequency dict."
    with gzip.open(filepath) as filehandle:
        return cpickle.load(filehandle)


def apply_filters(myvocab, setting='normal'):
    "Default setting of chained filters for trend detection."
    if setting not in ('loose', 'normal', 'strict'):
        print('invalid setting:', setting)
        setting = 'normal'
    for wordform in sorted(combined_filters(myvocab, setting)):
        print(wordform)


def calculate_bins(oldestday, newestday, interval=7):
    "Calculate time frame bins to fit the data (usually weeks)."
    return [d for d in range(oldestday, newestday - 1, -1) if oldestday - d >= 7 and d % interval == 0]


def refine_frequencies(vocab, bins):
    "Adjust the frequencies to a time frame and remove superfluous words."
    deletions = []
    # remove occurrences that are out of bounds: no complete week
    for word in vocab:
        new_series = array('H', [d for d in vocab[word]['time_series'] if not d < bins[-1] and not d > bins[0]])
        if len(new_series) <= 1:
            deletions.append(word)
        else:
            vocab[word]['time_series'] = new_series
    # remove words with too little data
    for word in deletions:
        del vocab[word]
    return vocab


def compute_frequencies(vocab, bins):
    "Compute absolute frequencies of words."
    timeseries = [0] * len(bins)
    # frequency computations
    freqsum = sum(len(vocab[l]['time_series']) for l in vocab)
    for wordform in vocab:
        # parts per million
        vocab[wordform]['total'] = float('{0:.3f}'.format((len(vocab[wordform]['time_series']) / freqsum)*1000000))
        freqseries = []
        days = Counter(vocab[wordform]['time_series'])
        for i, split in enumerate(bins):
            if i != 0:
                total = sum(days[d] for d in days if bins[i-1] < d <= split)
            else:
                total = sum(days[d] for d in days if d <= split)
            freqseries.append(total)
            timeseries[i] += total
        vocab[wordform]['series_abs'] = array('H', reversed(freqseries))
        # spare memory
        vocab[wordform]['time_series'] = []
    return vocab, list(reversed(timeseries))


def combine_frequencies(vocab, bins, timeseries):
    "Compute relative frequencies and word statistics."
    for wordform in vocab:
        vocab[wordform]['series_rel'] = array('f')
        for i in range(len(bins)):
            try:
                vocab[wordform]['series_rel'].append((vocab[wordform]['series_abs'][i] / timeseries[i])*1000000)
            except ZeroDivisionError:
                vocab[wordform]['series_rel'].append(0.0)
        # take non-zero values and perform calculations
        series = [f for f in vocab[wordform]['series_rel'] if f != 0.0]
        # todo: skip if series too short
        vocab[wordform]['stddev'] = float('{0:.3f}'.format(np.std(series)))
        vocab[wordform]['mean'] = float('{0:.3f}'.format(np.mean(series)))
        # spare memory
        vocab[wordform]['series_abs'] = []
    return vocab


def gen_freqlist(mydir, langcodes=[], maxdiff=1000, mindiff=0):
    "Compute long-term frequency info out of a directory containing text files."
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
            elif timediff < newestday:
                newestday = timediff
            # extract
            for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
                if is_relevant_input(token) is True:
                    if token not in myvocab:
                        myvocab[token] = dict(time_series=array('H'))
                    myvocab[token]['time_series'].append(timediff)

    # lemmatize and dehyphen
    myvocab = refine_vocab(myvocab, lemmadata)

    # determine bins
    bins = calculate_bins(oldestday, newestday)
    if not bins:
        print('Not enough days to compute frequencies')
        return freqs

    # clean and refine the data
    myvocab = refine_frequencies(myvocab, bins)

    # frequency computations
    myvocab, timeseries = compute_frequencies(myvocab, bins)

    # sum up frequencies
    myvocab = combine_frequencies(vocab, timeseries)

    return myvocab


def store_freqlist(freqs, filename, thres_a=1, thres_b=0.2):
    "Write relevant (defined by frequency) long-term occurrences info to a file."
    with open(filename, 'w') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'total', 'mean', 'stddev', 'relfreqs'])
        for entry in sorted(freqs):
            # only store statistically significant entries
            if freqs[entry]['stddev'] == 0:
                continue
            if freqs[entry]['mean'] > thres_a or \
                (
                    freqs[entry]['mean'] > thres_b and \
                    freqs[entry]['stddev'] < freqs[entry]['mean']/2
                ):
                tsvwriter.writerow(
                    [entry, freqs[entry]['total'], freqs[entry]['mean'],
                     freqs[entry]['stddev'], freqs[entry]['series_rel']]
                )


if __name__ == '__main__':
    print('Shoten.')

#!/usr/bin/env python

"""Tests for `shoten` package."""

import re
import sys
import tempfile

from array import array
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from shoten import *
from shoten.cli import parse_args, process_args
from shoten.filters import *

from simplemma import load_data


DE_LEMMADATA = load_data('de')


# Turns a dictionary into a class
class ToEntry(Entry):
    "Convert a dictionary to the entry class."
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
        setattr(self, 'series_rel', array('f'))


def test_basics():
    """Test basic functions."""
    # dates
    assert calc_timediff('2020 A') is None
    assert calc_timediff('2020-01-01') > 1
    assert calc_timediff('2030-01-01') < 1
    # load words
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3
    assert myvocab['Other'].sources == Counter({'Source1': 1})
    # pickling and unpickling
    os_handle, filepath = tempfile.mkstemp(suffix='.pickle', text=True)
    pickle_wordinfo(myvocab, filepath)
    myvocab2 = unpickle_wordinfo(filepath)
    assert len(myvocab2) == len(myvocab) and myvocab2['Tests'].time_series.all() == myvocab['Tests'].time_series.all()
    # generate from XML file
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir' / 'test2'), langcodes=('de'))
    assert len(myvocab) == 1 and 'Telegram' in myvocab
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), langcodes=('de', 'en'))
    assert 'Messengerdienst' in myvocab
    # without language codes and with short time frame
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), langcodes=[])
    assert 'Messengerdienst' in myvocab
    # without language codes and with short time frame
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), maxdiff=1)
    assert 'Messengerdienst' not in myvocab
    # with author filter
    myregex = re.compile(r'\b(|afp|apa|dpa)\b', re.I)
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), authorregex=myregex)
    assert 'Messengerdienst' not in myvocab
    # test frequency calculations
    assert gen_freqlist(str(Path(__file__).parent / 'testdir')) == {}
    assert gen_freqlist(str(Path(__file__).parent / 'testdir'), langcodes=['en']) == {}
    mydict = {
        'Test': ToEntry({'total': 20, 'mean': 10, 'stddev': 0, 'series_rel': [10, 10]})
    }
    # write to temp file
    os_handle, temp_outputfile = tempfile.mkstemp(suffix='.tsv', text=True)
    result = store_freqlist(mydict, temp_outputfile)
    assert result is None
    mydict['Test2'] = ToEntry({'total': 10, 'mean': 5, 'stddev': 4.082, 'series_rel': [10, 5, 0]})
    result = store_freqlist(mydict, temp_outputfile)
    assert result is None


def test_internals():
    """Test internal functions."""
    # filter known lemmata
    assert filter_lemmaform('Berge', DE_LEMMADATA, lemmafilter=True) is None
    assert filter_lemmaform('Berge', DE_LEMMADATA, lemmafilter=False) == 'Berg'

    # store in vocabulary
    myvocab = {}
    myvocab = putinvocab(myvocab, 'Berge', 5, 'Source_0', inheadings=True)
    assert 'Berge' in myvocab and myvocab['Berge'].time_series == array('H', [5]) and myvocab['Berge'].headings is True

    # de-hyphening
    myvocab = {
        'de-hyphening': ToEntry({'time_series': array('H', [1, 2, 3]), 'sources': Counter(['source1']), 'headings': True}),
        'dehyphening': ToEntry({'time_series': array('H', [3, 4]), 'sources': Counter(['source2']), 'headings': False})
    }
    newvocab = prune_vocab(deepcopy(myvocab), 'de-hyphening', 'dehyphening')
    assert len(newvocab['dehyphening'].time_series) == 5
    assert newvocab['dehyphening'].headings is True
    newvocab = dehyphen_vocab(deepcopy(myvocab))
    assert newvocab != myvocab
    assert len(newvocab['dehyphening'].time_series) == 5
    assert newvocab['dehyphening'].headings is True

    # refine vocab
    myvocab['Berge'] = ToEntry({'time_series': array('H', [5]), 'sources': Counter(['source3']), 'headings': False})
    newvocab = refine_vocab(deepcopy(myvocab), DE_LEMMADATA, lemmafilter=False, dehyphenation=False)
    assert 'Berg' in newvocab and 'de-hyphening' in newvocab
    newvocab = refine_vocab(deepcopy(myvocab), DE_LEMMADATA, lemmafilter=True, dehyphenation=True)
    assert 'Berge' not in newvocab and 'Berg' not in newvocab and 'de-hyphening' not in newvocab

    # filter levels
    newvocab = convert_to_numpy(newvocab)
    apply_filters(deepcopy(newvocab))
    apply_filters(newvocab, setting='loose')
    with pytest.raises(ZeroDivisionError):
        apply_filters(newvocab, setting='strict')

    # frequencies
    oldestday, newestday = 21, 1
    myvocab = {
        'Berg': ToEntry({'time_series': array('H', [newestday, 10, 10, oldestday])}),
        'Meer': ToEntry({'time_series': array('H', [newestday, 8, 9, 11, oldestday])}),
        'Tal': ToEntry({'time_series': array('H', [newestday, oldestday])}),
        'Zebra': ToEntry({'time_series': array('H', [10])})
    }
    bins = calculate_bins(oldestday, newestday)
    assert bins == [14, 7]
    myvocab = refine_frequencies(myvocab, bins)
    assert 'Berg' in myvocab and 'Tal' not in myvocab and 'Zebra' not in myvocab

    # freq calculations
    myvocab, timeseries = compute_frequencies(myvocab, bins)
    assert myvocab['Berg'].total == 400000.0 and myvocab['Berg'].series_abs == array('H', [0, 2])
    assert myvocab['Meer'].total == 600000.0 and myvocab['Meer'].series_abs == array('H', [0, 3])
    assert timeseries == [0, 5]
    myvocab = combine_frequencies(myvocab, bins, timeseries)
    assert myvocab['Berg'].series_rel == array('f', [0, 400000.0])
    assert myvocab['Meer'].series_rel == array('f', [0, 600000.0])


def test_cli():
    """Basic tests for command-line interface."""
    mydir = str(Path(__file__).parent / 'testdir')
    testargs = ['', '--read-dir', mydir, '-l', 'de', '--filter-level', 'loose']
    with patch.object(sys, 'argv', testargs):
        args = parse_args(testargs)
    process_args(args)
    myfile = str(Path(__file__).parent / 'testdir' / 'wordlist.tsv')
    testargs = ['', '--read-file', myfile, '-l', 'de', '--verbose']
    with patch.object(sys, 'argv', testargs):
        args = parse_args(testargs)
    process_args(args)


def test_filters():
    "Test functions for filtering of vocabulary."
    # init
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3

    # combined
    newvocab = combined_filters(deepcopy(myvocab), 'unknown')
    # assert newvocab == myvocab

    # input
    assert is_relevant_input('testing')
    assert is_relevant_input('testing.')
    assert not is_relevant_input('')
    assert not is_relevant_input('123')
    assert not is_relevant_input('abcde12345')
    assert not is_relevant_input('ABCDEF')

    # lemmata
    newvocab = recognized_by_simplemma(deepcopy(myvocab), DE_LEMMADATA)
    assert newvocab == {}

    # filters
    newvocab = sources_freqfilter(deepcopy(myvocab))
    assert len(newvocab) == 2
    newvocab = shortness_filter(deepcopy(newvocab))
    assert len(newvocab) == 1
    newvocab = sources_filter(deepcopy(myvocab), Counter(['Source1']))
    assert len(newvocab) == 1
    newvocab = wordlist_filter(deepcopy(myvocab), ['Tests', 'Other'], keep_words=False)
    assert len(newvocab) == 1
    newvocab = wordlist_filter(deepcopy(myvocab), ['Tests', 'Other'], keep_words=True)
    assert len(newvocab) == 2

    # scoring function
    scores = {}
    scores = scoring_func(scores, 1, newvocab)
    assert scores == {'Other': 1, 'Tests': 1}
    scores = scoring_func(scores, 1, newvocab)
    assert scores == {'Other': 2, 'Tests': 2}

    # hyphens
    myvocab = {
        'de-hyphening': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'hyphens-stuff': ToEntry({'time_series': array('H', [3, 3])}),
        'hyphen-stuff': ToEntry({'time_series': array('H', [3, 3, 3])}),
        'stuff': ToEntry({'time_series': array('H', [3, 3, 3, 3, 3])})
    }
    myvocab = convert_to_numpy(myvocab)
    newvocab = hyphenated_filter(myvocab, perc=0, verbose=False)
    assert list(newvocab.keys()) == ['de-hyphening', 'stuff']

    # zipf
    myvocab = {
        'short': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'the-longest-word': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'longer': ToEntry({'time_series': array('H', [1])}),
        'other1': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'this': ToEntry({'time_series': array('H', [1, 2])}),
        'one': ToEntry({'time_series': array('H', [1, 2, 3, 4, 5])}),
    }
    myvocab = convert_to_numpy(myvocab)
    newvocab = zipf_filter(deepcopy(myvocab), freqperc=80, lenperc=20, verbose=True)
    assert len(newvocab) == 5
    newvocab = zipf_filter(deepcopy(myvocab), freqperc=20, lenperc=20, verbose=True)
    assert len(newvocab) == 4

    # headings
    myvocab = {
        'Berg': ToEntry({'time_series': array('H', [1, 2, 3]), 'headings': True}),
        'Tal': ToEntry({'time_series': array('H', [1, 2, 3]), 'headings': False}),
    }
    myvocab = headings_filter(myvocab)
    assert len(myvocab) == 1

    # n-grams
    myvocab = dict.fromkeys(['Berg', 'Berge', 'Bergen', 'Berger', 'Bernstein', 'Tal', 'Täler'])
    newvocab = ngram_filter(deepcopy(myvocab), threshold=50, verbose=True)
    assert list(newvocab.keys()) == ['Berg', 'Berge', 'Berger', 'Bernstein', 'Tal']
    newvocab = ngram_filter(deepcopy(myvocab), threshold=10, verbose=True)
    assert list(newvocab.keys()) == ['Berge', 'Bernstein', 'Tal']



#def test_readme():
#    """Test function to verify readme examples."""
#    # ...
#

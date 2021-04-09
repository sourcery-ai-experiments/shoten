#!/usr/bin/env python

"""Tests for `shoten` package."""

import re
import pytest
import sys

from pathlib import Path
from unittest.mock import patch

from shoten import *
from shoten.cli import parse_args, process_args
from shoten.filters import *



def test_basics():
    """Test basic functions."""
    assert calc_timediff('2020 A') is None
    assert calc_timediff('2020-01-01') > 1
    assert calc_timediff('2030-01-01') < 1
    # load words
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3
    assert myvocab['Other']['sources'] == ['Source1']
    # pickling and unpickling
    filepath = str(Path(__file__).parent / 'test.pickle')
    pickle_wordinfo(myvocab, filepath)
    myvocab2 = unpickle_wordinfo(filepath)
    assert len(myvocab2) == len(myvocab) and myvocab2['Tests']['time_series'].all() == myvocab['Tests']['time_series'].all()
    # generate from XML file
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


def test_cli():
    """Basic tests for command-line interface."""
    mydir = str(Path(__file__).parent / 'testdir')
    testargs = ['', '--read-dir', mydir, '-l', 'de', '--filter-level', 'loose']
    with patch.object(sys, 'argv', testargs):
        args = parse_args(testargs)
    process_args(args)


def test_filters():
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3
    myvocab = sources_freqfilter(myvocab)
    assert len(myvocab) == 2
    myvocab = shortness_filter(myvocab)
    assert len(myvocab) == 1
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    myvocab = sources_filter(myvocab, {'Source1'})
    assert len(myvocab) == 1
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    myvocab = wordlist_filter(myvocab, ['Tests', 'Other'], keep_words=False)
    assert len(myvocab) == 1
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    myvocab = wordlist_filter(myvocab, ['Tests', 'Other'], keep_words=True)
    assert len(myvocab) == 2
    


#def test_readme():
#    """Test function to verify readme examples."""
#    # ...
#

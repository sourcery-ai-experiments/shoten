#!/usr/bin/env python

"""Tests for `shoten` package."""

import pytest
import sys

from pathlib import Path
from unittest.mock import patch

from shoten import *
from shoten.cli import parse_args, process_args



def test_basics():
    """Test basic functions."""
    assert calc_timediff('2020 A') is None
    assert calc_timediff('2020-01-01') > 1
    assert calc_timediff('2030-01-01') < 1
    # load words
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 2
    # pickling and unpickling
    filepath = str(Path(__file__).parent / 'test.pickle')
    pickle_wordinfo(myvocab, filepath)
    myvocab2 = unpickle_wordinfo(filepath)
    assert len(myvocab2) == len(myvocab) and myvocab2['Tests'].all() == myvocab['Tests'].all()
    # generate from XML file
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), ('de', 'en'))
    assert 'Messengerdienst' in myvocab
    # without language codes
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), [])
    assert 'Messengerdienst' in myvocab


def test_cli():
    """Basic tests for command-line interface."""
    mydir = str(Path(__file__).parent / 'testdir')
    testargs = ['', '--read-dir', mydir, '-l', 'de', '--filter-level', 'loose']
    with patch.object(sys, 'argv', testargs):
        args = parse_args(testargs)
    process_args(args)
    


#def test_readme():
#    """Test function to verify readme examples."""
#    # ...
#

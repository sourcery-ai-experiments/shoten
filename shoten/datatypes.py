"""Module listing data classes, types and constants used by shoten."""


#from __future__ import annotations

from array import array
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Dict, Iterator, List, Union

# Python 3.7+
# from numpy.typing import ArrayLike  # type: ignore[import]


TODAY = datetime.now()


ARRAY_TYPE = 'H'
MAX_SERIES_VAL = 65535

MAX_NGRAM_VOC = 15000


class Entry:
    "Defines a class for dictionaries entries, containing metadata and stats."
    __slots__ = ['absfreq', 'headings', 'mean', 'series_abs', 'series_rel', 'sources', 'stddev', 'time_series', 'total']
    def __init__(self) -> None:
        self.absfreq: int
        self.headings: bool = False
        self.mean: float
        self.series_abs: Any = array('f')
        self.series_rel: Any = array('f')
        self.sources: Dict[str, int] = defaultdict(int)
        self.stddev: float
        self.time_series: Dict[int, int] = defaultdict(int)
        self.total: float


def dict_sum(one: Dict[Any, int], two: Dict[Any, int]) -> Dict[Any, int]:
    "Add up two dictionaries, fast."
    return {k: one.get(k, 0) + two.get(k, 0) for k in set(one) | set(two)}


def flatten_series(entry: Entry) -> Union[Iterator[int], List[int]]:
    "Flatten a defaultdict(int) to a list."
    mydict = entry.time_series
    # optimized, returns list
    if len(mydict) < 50:
        mylist = []
        _ = [mylist.extend([k]*v) for k, v in mydict.items()]  # type: ignore[func-returns-value]
        return mylist
    # computationally stable for larger dicts, returns iterator
    return Counter(mydict).elements()


def sum_entry(entry: Entry) -> int:
    "Get the total number of occurrences of a word by summing the values in its dict."
    return sum(entry.time_series.values())


def sum_entries(vocab: Dict[str, Entry]) -> List[int]:
    "Return all frequencies in the vocabulary by word."
    return [sum_entry(e) for e in vocab.values()]

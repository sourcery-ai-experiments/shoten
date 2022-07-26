"""Module listing data classes, types and constants used by shoten."""


#from __future__ import annotations

from array import array
from collections import defaultdict  # Counter
from datetime import datetime
from typing import Any, Dict

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
        self.series_abs: Any[int] = array('f')
        self.series_rel: Any[float] = array('f')
        self.sources: Dict[str, int] = defaultdict(int)
        self.stddev: float
        self.time_series: Any[int] = array(ARRAY_TYPE)
        self.total: float

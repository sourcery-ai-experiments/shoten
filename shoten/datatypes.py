"""Module listing data classes, types and constants used by shoten."""


#from __future__ import annotations

from array import array
from collections import defaultdict  # Counter
from datetime import datetime
from typing import Dict, Union

try:
    from numpy.typing import ArrayLike as NDArray # type: ignore[import]
except ImportError:
    from nptyping import NDArray  # type: ignore[import]


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
        self.series_abs: NDArray[int] = array('f')
        self.series_rel: NDArray[float] = array('f')
        self.sources: Dict[str, int] = defaultdict(int)
        self.stddev: float
        self.time_series: NDArray[int] = array(ARRAY_TYPE)
        self.total: float

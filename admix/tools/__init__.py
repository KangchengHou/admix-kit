"""
External tools
"""

from dask.array.core import Array
import dask.array as da
from ._utils import get_dependency, get_cache_data
from . import gcta, plink2, liftover, plink
from ._standalone import hapgen2, admix_simu, haptools_simu_admix


__all__ = [
    "plink2",
    "gcta",
    "lift_over",
    "plink_assoc",
    "plink_read_bim",
    "plink_read_fam",
]

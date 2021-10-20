import pandas as pd
import xarray as xr
import numpy as np
import dask.array as da
from typing import (
    List,
    Union,
    Any,
    Dict,
    Sequence,
)

# convert the indexer (integer, slice, string, array) to the actual positions
# reference: https://github.com/theislab/anndata/blob/566f8fe56f0dce52b7b3d0c96b51d22ea7498156/anndata/_core/index.py#L16
def _normalize_index(
    indexer: Union[
        slice,
        int,
        str,
        np.ndarray,
    ],
    index: pd.Index,
) -> Union[slice, int, np.ndarray]:  # ndarray of int
    if not isinstance(index, pd.RangeIndex):
        assert (
            index.dtype != float and index.dtype != int
        ), "Don’t call _normalize_index with non-categorical/string names"

    # the following is insanely slow for sequences,
    # we replaced it using pandas below
    def name_idx(i):
        if isinstance(i, str):
            i = index.get_loc(i)
        return i

    if isinstance(indexer, slice):
        start = name_idx(indexer.start)
        stop = name_idx(indexer.stop)
        # string slices can only be inclusive, so +1 in that case
        if isinstance(indexer.stop, str):
            stop = None if stop is None else stop + 1
        step = indexer.step
        return slice(start, stop, step)
    elif isinstance(indexer, (np.integer, int)):
        return indexer
    elif isinstance(indexer, str):
        return index.get_loc(indexer)  # int
    elif isinstance(indexer, (Sequence, np.ndarray, pd.Index, np.matrix)):
        if hasattr(indexer, "shape") and (
            (indexer.shape == (index.shape[0], 1))
            or (indexer.shape == (1, index.shape[0]))
        ):
            indexer = np.ravel(indexer)
        if not isinstance(indexer, (np.ndarray, pd.Index)):
            indexer = np.array(indexer)
        if issubclass(indexer.dtype.type, (np.integer, np.floating)):
            return indexer  # Might not work for range indexes
        elif issubclass(indexer.dtype.type, np.bool_):
            if indexer.shape != index.shape:
                raise IndexError(
                    f"Boolean index does not match Dataset’s shape along this "
                    f"dimension. Boolean index has shape {indexer.shape} while "
                    f"Dataset index has shape {index.shape}."
                )
            positions = np.where(indexer)[0]
            return positions  # np.ndarray[int]
        else:  # indexer should be string array
            positions = index.get_indexer(indexer)
            if np.any(positions < 0):
                not_found = indexer[positions < 0]
                raise KeyError(
                    f"Values {list(not_found)}, from {list(indexer)}, "
                    "are not valid obs/ var names or indices."
                )
            return positions  # np.ndarray[int]
    else:
        raise IndexError(f"Unknown indexer {indexer!r} of type {type(indexer)}")


def unpack_index(index):
    if not isinstance(index, tuple):
        return index, slice(None)
    elif len(index) == 2:
        return index
    elif len(index) == 1:
        return index[0], slice(None)
    else:
        raise IndexError("invalid number of indices")


def normalize_indices(index, snp_names, indiv_names):

    # deal with tuples of length 1
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]

    if isinstance(index, tuple):
        if len(index) > 2:
            raise ValueError("AnnData can only be sliced in rows and columns.")

    snp_ax, indiv_ax = unpack_index(index)
    snp_ax = _normalize_index(snp_ax, snp_names)
    indiv_ax = _normalize_index(indiv_ax, indiv_names)
    return snp_ax, indiv_ax
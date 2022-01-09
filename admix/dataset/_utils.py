import pandas as pd
import numpy as np
import admix
from typing import (
    List,
    Union,
    Tuple,
    Sequence,
)


def all_array_equal(array_list: List[np.ndarray]) -> bool:
    """check whether all arrays in the list are equal

    Parameters
    ----------
    List[np.ndarray]

    Returns
    -------
    bool
    """
    return all([np.allclose(array_list[0], a) for a in array_list[1:]])


def align_datasets(dsets: List[admix.Dataset], dim: str) -> List[admix.Dataset]:
    """takes 2 or more datasets, return the aligned dataset

    SNP alignment is based on index, CHROM, POS, REF, ALT
    Individual alignment is based on index

    TODO: refer to PLINK2 alignment page for preprocessing your data to align SNP attributes
    before hand.

    Parameters
    ----------
    dsets : List[Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    List[admix.Dataset]: list of aligned datasets
    """
    assert dim in ["snp", "indiv"]
    if dim == "snp":
        required_snp_columns = admix.dataset.REQUIRED_SNP_COLUMNS
        # TODO: find aligned snp index among datasets


def check_align(dsets: List[admix.Dataset], dim: str) -> bool:
    """takes 2 or more datasets, and check whether the common attributes among datasets
    are properly aligned

    Parameters
    ----------
    dsets : List[admix.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    bool: whether the two datasets align in the given dimension
    """
    aligned = False
    assert dim in ["snp", "indiv"], "dim must be either 'snp' or 'indiv'"
    if dim == "snp":
        required_snp_columns = admix.dataset.REQUIRED_SNP_COLUMNS

        # find common columns among datasets
        common_cols = set.intersection(*[set(dset.snp.columns) for dset in dsets])

        # common_cols must be a subset of required_snp_columns
        assert set(required_snp_columns).issubset(
            common_cols
        ), "common_cols must be a subset of required_snp_columns"

        inconsistent_cols = [
            col
            for col in common_cols
            if not all_array_equal([dset.snp[col].values for dset in dsets])
        ]
        if not all_array_equal([dset.snp.index.values for dset in dsets]):
            inconsistent_cols.append("index")
        if len(inconsistent_cols) > 0:
            print(f"inconsistent columns: {','.join(inconsistent_cols)} in snp")
            aligned = False
        else:
            aligned = True

    elif dim == "indiv":
        # find common columns among datasets
        common_cols = set.intersection(*[set(dset.indiv.columns) for dset in dsets])
        # check whether all datasets have the same columns
        inconsistent_cols = [
            col
            for col in common_cols
            if not all_array_equal([dset.indiv[col].values for dset in dsets])
        ]
        if len(inconsistent_cols) > 0:
            print(f"inconsistent columns: {','.join(inconsistent_cols)} in indiv")
            aligned = False
        else:
            aligned = True
    return aligned

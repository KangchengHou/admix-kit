from typing import List
import xarray as xr


def check_align(dsets: List[xr.Dataset], dim: str) -> bool:
    """takes 2 or more datasets, and check whether attributes align

    Parameters
    ----------
    dsets : List[xr.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    bool: whether the two datasets align in the given dimension
    """
    assert dim in ["snp", "indiv"]
    pass


def align(dsets: List[xr.Dataset], dim: str) -> List[xr.Dataset]:
    """takes 2 or more datasets, return the aligned dataset

    Parameters
    ----------
    dsets : List[xr.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    List[xr.Dataset]: list of aligned datasets
    """
    assert dim in ["snp", "indiv"]
    pass


__all__ = ["check_align", "align"]
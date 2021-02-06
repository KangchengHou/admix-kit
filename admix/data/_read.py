import pandas as pd
from pysnptools.snpreader import Bed
import numpy as np
from ._utils import read_int_mat, write_int_mat

def read_lanc(path: str) -> np.ndarray:
    """Read local ancestry

    Args:
        path (str): path to the local ancestry

    Returns:
        np.ndarray: (n_haplo, n_snp) local ancestry file
    """
    return read_int_mat(path)

def read_haplo(path: str) -> np.ndarray:
    """Read haplotypes

    Args:
        path (str): path to the haplotypes

    Returns:
        np.ndarray: (n_haplo, n_snp) local ancestry file
    """
    return read_int_mat(path)
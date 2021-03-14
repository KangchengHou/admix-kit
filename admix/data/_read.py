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


def read_hap(path: str) -> np.ndarray:
    """Read haplotypes

    Args:
        path (str): path to the haplotypes

    Returns:
        np.ndarray: (n_haplo, n_snp) local ancestry file
    """
    return read_int_mat(path)


def read_geno(prefix: str):
    """Read genotype

    Parameters
    ----------
    path : str
        path to the genotype

    Returns
    -------
    [type]
        [description]
    """
    hap = read_int_mat(prefix + ".hap.gz")
    legend = pd.read_csv(prefix + ".legend.gz", delim_whitespace=True)
    sample = pd.read_csv(prefix + ".sample.gz", delim_whitespace=True)
    return {"hap": hap, "legend": legend, "sample": sample}

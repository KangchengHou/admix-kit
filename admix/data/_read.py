import pandas as pd
from pysnptools.snpreader import Bed
import numpy as np
from ._genotype import Genotype
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

    
def read_plink(path: str) -> Genotype:
    """Read plink file as a genotype object

    Args:
        path (str): the path to the plink file without suffix

    Returns:
        Genotype: the genotype project
    """
    data = Bed(path)
    
    return Genotype(X=None, indiv=None, snp=None)
    
    


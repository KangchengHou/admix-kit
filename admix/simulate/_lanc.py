from scipy.optimize import fsolve
import numpy as np
from scipy.special import logit, expit
from typing import List


def simulate_lanc(n_hap: int, n_snp: int, chunk_size: int) -> np.ndarray:
    """Simulate local ancestry

    Parameters
    ----------
    n_hap : int
        Number of haplotypes
    n_snp : int
        Number of SNPs
    chunk_size : int
        Size of mosaic chunk

    Returns
    -------
    np.ndarray
        Simulated local ancestry
    """
    lanc = np.zeros((n_hap, n_snp), dtype=int)
    start_loc = np.random.randint(chunk_size, size=n_hap)
    for i in range(n_hap):
        n_chunk = int(n_snp / chunk_size)
        this_local_anc = np.concatenate(
            [
                np.zeros(start_loc[i]),
                np.repeat(np.mod(np.arange(n_chunk) + 1, 2), chunk_size),
            ]
        )
        lanc[i, :] = this_local_anc[0:n_snp]
    return lanc
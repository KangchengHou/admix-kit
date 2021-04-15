import numpy as np


def simulate_hap(n_hap: int, n_snp: int) -> np.ndarray:
    """Simulate haplotype

    Parameters
    ----------
    n_hap :
        Number of haplotypes
    n_snp : [type]
        Number of SNPs

    Returns
    -------
        np.ndarray
        Simulated haplotypes
    """
    return np.random.randint(2, size=(n_hap, n_snp), dtype=int)

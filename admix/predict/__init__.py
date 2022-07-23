import numpy as np
import pandas as pd
import statsmodels.api as sm
import admix


def partial_pgs(dset: admix.Dataset, weight: np.ndarray):
    """
    Calculate partial PGS for admix individuals

    Parameters
    ----------
    dset : admix.Dataset
        Dataset object
    weight : np.ndarray
        PGS weights

    Returns
    -------
    pgs : np.ndarray
        Partial PGS for each individual

    References
    ----------
    .. [1] Marnetto et al. (2020) Nat Commun.
    """
    pass
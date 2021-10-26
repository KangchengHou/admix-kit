import numpy as np
from scipy import stats


def pval2chisq(pval: np.ndarray, two_sided: bool = True):
    """Convert p-value to z-score

    Parameters
    ----------
    pval : np.ndarray
        p-value
    two_sided : bool
        Whether to use two-sided test

    Returns
    -------
    np.ndarray
        z-score
    """
    if two_sided:
        return stats.norm.ppf(pval / 2) ** 2
    else:
        return stats.norm.ppf(pval) ** 2


def zsc2pval(zscore: np.ndarray, two_sided: bool = True):
    """Convert z-score to p-value

    Parameters
    ----------
    zscore : np.ndarray
        z-score
    two_sided : bool
        Whether to use two-sided test

    Returns
    -------
    np.ndarray
        p-value
    """
    if two_sided:
        return 2 * (1 - stats.norm.cdf(np.abs(zscore)))
    else:
        return 1 - stats.norm.cdf(zscore)


def quantile_normalize(val):
    from scipy.stats import rankdata, norm

    val = np.array(val)
    non_nan_index = ~np.isnan(val)
    results = np.full(val.shape, np.nan)
    results[non_nan_index] = norm.ppf(
        (rankdata(val[non_nan_index]) - 0.5) / len(val[non_nan_index])
    )
    return results

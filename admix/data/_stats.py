import numpy as np
from scipy import stats


def lambda_gc(pval, bootstrap_ci=False, n_resamples=499):
    def _lambda(pval):
        chi2 = stats.norm.ppf(pval / 2) ** 2
        return np.quantile(chi2, 0.5) / stats.chi2.ppf(0.5, 1)

    from scipy.stats import bootstrap

    est = _lambda(pval)
    if bootstrap_ci:
        res = bootstrap(
            (pval,),
            _lambda,
            axis=-1,
            vectorized=False,
            n_resamples=n_resamples,
        )
        ci = res.confidence_interval
        return est, (ci[0], ci[1])
    else:
        return est


def pval2chisq(pval: np.ndarray, two_sided: bool = True):
    """Convert p-value to chisq

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
    pval = np.array(pval)
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

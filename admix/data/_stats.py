import numpy as np
from scipy import stats
import scipy
from typing import Union, List, Tuple


def hdi(
    x: np.ndarray, loglik: np.ndarray, ci: float = 0.95
) -> Union[List[Tuple], Tuple]:
    """
    Find the high density interval for 1-dimensional likelihood curve.

    Parameters
    ----------
    x : np.ndarray
        1-dimensional data
    loglik : np.ndarray
        log-likelihood of the corresponding data
    ci : float
        targeted interval

    Returns
    -------
    Tuple, List[Tuple]
        High density interval for the data. Returns a tuple denoting [low_ci, high_ci].
        Returns a list of tuples in cases where the likelihood curve is not concave (
        typically indicating the low sample size when maximizing the likelihood).
    """
    prob = np.exp(loglik - np.max(loglik))
    prob /= prob.sum()
    sorted_prob = np.sort(prob)[::-1]
    # critical value
    crit = sorted_prob[np.argmax(np.cumsum(sorted_prob) >= ci)]
    np.where(prob > crit)
    hdi_index = np.where(prob > crit)[0]

    from itertools import groupby
    from operator import itemgetter

    intervals: List[Tuple] = []
    for k, g in groupby(enumerate(hdi_index), lambda ix: ix[0] - ix[1]):
        hdi_index = list(map(itemgetter(1), g))
        intervals.append((x[hdi_index[0]], x[hdi_index[-1]]))
    if len(intervals) == 1:
        return intervals[0]
    else:
        return intervals


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


def pval2zsc(pval):
    return -scipy.stats.norm.ppf(pval)


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


def quad_form(x, A):
    return np.dot(np.dot(x.T, A), x)


def chi2_to_logpval(chi2, dof=1):
    return scipy.stats.chi2.logsf(chi2, dof)


def deming_regression(
    x: np.ndarray,
    y: np.ndarray,
    sx: np.ndarray = None,
    sy: np.ndarray = None,
    no_intercept: bool = False,
):
    """Deming regression.

    Parameters
    ----------
    x : np.ndarray
        x variables
    y : np.ndarray
        y variables
    sx : np.ndarray, optional
        standard errors of x variables, by default None
    sy : np.ndarray, optional
        standard errors of y variables, by default None
    no_intercept : bool, optional
        whether not to fit intercept or not, by default False

    Returns
    -------
    If no_intercept is False:
        no intercept is fit, return a single slope
    If no_intercept is True:
        intercept is fit, return slope and intercept
    """

    def no_intercept_func(B, x):
        return B[0] * x

    if no_intercept:
        model = scipy.odr.Model(no_intercept_func)
        odr = scipy.odr.ODR(scipy.odr.RealData(x, y, sx=sx, sy=sy), model, beta0=[1])
        fit = odr.run()
        return fit.beta[0]
    else:
        model = scipy.odr.unilinear
        odr = scipy.odr.ODR(scipy.odr.RealData(x, y, sx=sx, sy=sy), model)
        fit = odr.run()
        return fit.beta[0], fit.beta[1]


def meta_analysis(
    effects: np.ndarray, se: np.ndarray, method="random", weights: np.ndarray = None
) -> float:
    """Meta analysis of effects

    Parameters
    ----------
    effects : np.ndarray
        effects array
    se : np.ndarray
        effects array
    method : str, optional
        method for meta-analysis, by default "random"
    weights : np.ndarray, optional
        weight for different effects, by default None

    Returns
    -------
    float
        single number summarizing the meta-analysis results
    """
    # From Omer Weissbrod
    assert method in ["fixed", "random"]
    d = effects
    variances = se ** 2

    # compute random-effects variance tau2
    vwts = 1.0 / variances
    fixedsumm = vwts.dot(d) / vwts.sum()
    Q = np.sum(((d - fixedsumm) ** 2) / variances)
    df = len(d) - 1
    tau2 = np.maximum(0, (Q - df) / (vwts.sum() - vwts.dot(vwts) / vwts.sum()))

    # defing weights
    if weights is None:
        if method == "fixed":
            wt = 1.0 / variances
        else:
            wt = 1.0 / (variances + tau2)
    else:
        wt = weights

    # compute summtest
    summ = wt.dot(d) / wt.sum()
    if method == "fixed":
        varsum = np.sum(wt * wt * variances) / (np.sum(wt) ** 2)
    else:
        varsum = np.sum(wt * wt * (variances + tau2)) / (np.sum(wt) ** 2)

    summary = summ
    se_summary = np.sqrt(varsum)

    return summary, se_summary
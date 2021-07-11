import numpy as np
from scipy import stats
from scipy.special import logsumexp


def pval_to_posterior(pval):
    """
    Convert marginal p-value to posterior probability

    Args
    -----
    pval: (n_snp, )
    """
    zsc = stats.norm.ppf(pval / 2)
    logprob = -stats.norm.logpdf(zsc)
    return np.exp(logprob - logsumexp(logprob))


def posterior_to_credible_set(posterior, coverage=0.9):
    """
    From the given posterior probability to the credible set

    Args
    -----
    posterior: (n_snp, )
    coverage: desired coverage
    """
    n_cs = np.sum(np.cumsum(np.sort(posterior)[::-1]) < coverage) + 1
    return np.argsort(posterior)[::-1][0:n_cs]


def chi2_to_posterior(chi2, df):
    """
    Convert marginal p-value to the posterior probability of being causal

    Args
    -----
    pval: (n_snp, )
    df: degree of freedom
    """
    pval = stats.chi2.sf(chi2, df=df)
    return pval_to_posterior(pval)
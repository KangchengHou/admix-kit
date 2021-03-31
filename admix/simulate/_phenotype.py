from typing import Union
from scipy.optimize import fsolve
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from scipy.special import logit, expit
from ..data import convert_anc_count
from typing import List


def sample_case_control(pheno: np.ndarray, control_ratio: float) -> np.ndarray:
    """Sample case control from the population with a desired ratio

    Args:
        pheno (np.ndarray): (n_indiv, ) binary vector representing the case control status
            for each individual
        control_ratio (float): the ratio of control / case

    Returns:
        np.ndarray: (n_indiv, ) vector indicating whether i-th individual is sampled
    """
    case_index = np.where(pheno == 1)[0]
    control_index = np.random.choice(
        np.where(pheno == 0)[0],
        size=int(len(case_index) * control_ratio),
        replace=False,
    )
    study_index = np.sort(np.concatenate([case_index, control_index]))
    return study_index


def simulate_phenotype_case_control_1snp(
    hap: np.ndarray,
    lanc: np.ndarray,
    case_prevalence: float,
    odds_ratio: float,
    ganc: np.ndarray = None,
    ganc_effect: float = None,
    n_sim: int = 10,
) -> List[np.ndarray]:
    """Simulate case control phenotypes from phased genotype and ancestry (one SNP at a time)

    Args:
        hap (np.ndarray): phased genotype (n_indiv, 2 * n_snp), the first `n_snp` elements are
            for the first ancestry, the second are for the second ancestry
        lanc (np.ndarray): local ancestry (n_indiv, 2 * n_snp), same as `hap`
        case_prevalence (float): case prevalence in the population
        odds_ratio (float): odds ratio for the causal SNP
        ganc (np.ndarray, optional): (n_indiv, ) global ancestry. Defaults to None.
        ganc_effect (float, optional): (n_indiv, ) Effect of the global ancestry. Defaults to None.
        n_sim (int, optional): Number of simulations. Defaults to 10.

    Returns:
        List[np.ndarray]: `n_snp`-length of numpy array with shape (n_indiv, n_sim)
    """

    n_indiv = hap.shape[0]
    n_snp = hap.shape[1] // 2
    if ganc is not None:
        assert len(ganc) == n_indiv

    rls_list = []
    # simulate snp by snp
    for snp_i in range(n_snp):
        snp_hap = hap[:, [snp_i, snp_i + n_snp]]
        snp_lanc = lanc[:, [snp_i, snp_i + n_snp]]
        # the number of minor alleles at each location by ancestry
        snp_cnt = convert_anc_count(snp_hap, snp_lanc)

        # allelic risk effect size x number of minor alleles
        snp_phe_g = np.dot(snp_cnt, np.log(odds_ratio) * np.ones((2, n_sim)))
        if ganc is not None:
            snp_phe_g += np.dot(ganc[:, np.newaxis], ganc_effect * np.ones((1, n_sim)))
        snp_phe = np.zeros_like(snp_phe_g, dtype=np.int8)

        for sim_i in range(n_sim):
            # find an intercept, such that the expectation is case_prevalence.
            func = lambda b: np.mean(expit(b + snp_phe_g[:, sim_i])) - case_prevalence
            intercept = fsolve(func, logit(case_prevalence))
            snp_phe[:, sim_i] = np.random.binomial(
                1, expit(intercept + snp_phe_g[:, sim_i])
            )
        rls_list.append(snp_phe)
    return rls_list


def simulate_phenotype_continuous(
    hap: np.ndarray,
    lanc: np.ndarray,
    h2g: float,
    n_causal: int = None,
    ganc: np.ndarray = None,
    ganc_effect: float = None,
    cov=0.0,
    n_sim=10,
):
    """Simulate phenotype for admixture population [continuous]

    Parameters
    ----------
    hap : np.ndarray
        phased genotype (n_indiv, 2 * n_snp), the first `n_snp` elements are
        for the first haplotype, the second are for the second haplotype
    lanc : np.ndarray
        local ancestry (n_indiv, 2 * n_snp), same as `hap`
    h2g : float
        desired heritability
    n_causal : int, optional
        number of causal variables, by default None
    ganc : np.ndarray, optional
        vector of global ancestry, by default None
    ganc_effect : float, optional
        global ancestry effect, by default None
    cov : float, optional
        covariance of genetic effect, by default 0.0
    n_sim : int, optional
        number of simulations, by default 30

    Returns
    -------
    beta : np.ndarray
        simulated effect sizes (2 * n_snp, n_sim)
    phe_g: np.ndarray
        simulated genetic component of phenotypes (n_indiv, n_sim)
    phe: np.ndarray
        simulated phenotype (n_indiv, n_sim)
    """

    geno = convert_anc_count(hap, lanc)
    n_indiv = geno.shape[0]
    n_snp = geno.shape[1] // 2

    if ganc is not None:
        assert len(ganc) == n_indiv

    beta1 = np.zeros((n_snp, n_sim))
    beta2 = np.zeros((n_snp, n_sim))
    for i_sim in range(n_sim):
        cau = sorted(np.random.choice(np.arange(n_snp), size=n_causal, replace=False))
        beta = np.random.multivariate_normal(
            mean=[0.0, 0.0], cov=[[1.0, cov], [cov, 1.0]], size=n_causal
        )
        beta1[cau, i_sim] = beta[:, 0]
        beta2[cau, i_sim] = beta[:, 1]

    beta = np.vstack([beta1, beta2])

    phe_g = np.dot(geno, beta)
    phe_e = np.zeros_like(phe_g)

    for sim_i in range(n_sim):
        var_g = np.var(phe_g[:, sim_i])
        var_e = var_g * ((1.0 / h2g) - 1)
        phe_e[:, sim_i] = np.random.normal(loc=0.0, scale=np.sqrt(var_e), size=n_indiv)

    phe = phe_g + phe_e
    if ganc is not None:
        phe += np.dot(ganc[:, np.newaxis], ganc_effect * np.ones((1, n_sim)))
    return beta, phe_g, phe

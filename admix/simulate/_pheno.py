import numpy as np
from typing import List
import dask.array as da
import xarray as xr
from typing import Union, List, Dict
import admix
import dask
import pandas as pd
from tqdm import tqdm


def quant_pheno(
    dset: admix.Dataset,
    hsq: float = 0.5,
    cor: float = 1,
    n_causal: int = None,
    beta: np.ndarray = None,
    cov_cols: List[str] = None,
    cov_effects: List[float] = None,
    n_sim=10,
) -> dict:
    """Simulate continuous phenotype of admixed individuals [continuous]

    Parameters
    ----------
    dset: xr.Dataset
        Dataset containing the following variables:
            - geno: (n_indiv, n_snp, 2) phased genotype of each individual
            - lanc: (n_indiv, n_snp, 2) local ancestry of each SNP
    hsq: float
        Variance explained by the genotype effect
    cor: float
        Correlation between the genetic effects from two ancestral backgrounds
    n_causal: int, optional
        number of causal variables, by default None
    beta: np.ndarray, optional
        causal effect of each causal variable, by default None
    cov_cols: List[str], optional
        list of covariates to include as covariates, by default None
    cov_effects: List[float], optional
        list of the effect of each covariate, by default None
        for each simulation, the cov_effects will be the same
    n_sim : int, optional
        number of simulations, by default 10


    Returns
    -------
    beta: np.ndarray
        simulated effect sizes (2 * n_snp, n_sim)
    phe_g: np.ndarray
        simulated genetic component of phenotypes (n_indiv, n_sim)
    phe: np.ndarray
        simulated phenotype (n_indiv, n_sim)
    """
    n_anc = dset.n_anc
    assert n_anc == 2, "Only two-ancestry currently supported"

    dset.compute_allele_per_anc(center=False)

    apa = dset.allele_per_anc
    n_snp, n_indiv = apa.shape[0:2]

    # simulate effect sizes
    if beta is None:

        if n_causal is None:
            # n_causal = n_snp if `n_causal` is not specified
            n_causal = n_snp

        # if `beta` is not specified, simulate effect sizes
        beta = np.zeros((n_snp, n_anc, n_sim))
        for i_sim in range(n_sim):
            cau = sorted(
                np.random.choice(np.arange(n_snp), size=n_causal, replace=False)
            )

            i_beta = np.random.multivariate_normal(
                mean=[0.0, 0.0],
                cov=np.array([[1, cor], [cor, 1]]),
                size=n_causal,
            )

            for i_anc in range(n_anc):
                beta[cau, i_anc, i_sim] = i_beta[:, i_anc]
    else:
        assert (
            (hsq is None) and (cor is None) and (n_causal is None)
        ), "If `beta` is specified, `var_g`, `var_e`, `gamma`, and `n_causal` must be specified"
        assert beta.shape == (n_snp, n_anc) or beta.shape == (
            n_snp,
            n_anc,
            n_sim,
        ), "`beta` must be of shape (n_snp, n_anc) or (n_snp, n_anc, n_sim)"
        if beta.shape == (n_snp, n_anc):
            # replicate `beta` for each simulation
            beta = np.repeat(beta[:, :, np.newaxis], n_sim, axis=2)

    pheno_g = np.zeros([n_indiv, n_sim])
    snp_chunks = apa.chunks[0]
    indices = np.insert(np.cumsum(snp_chunks), 0, 0)

    for i in tqdm(range(len(indices) - 1), desc="admix.simulate.quant_pheno"):
        start, stop = indices[i], indices[i + 1]
        apa_chunk = apa[start:stop, :, :].compute()
        for i_anc in range(n_anc):
            pheno_g += np.dot(apa_chunk[:, :, i_anc].T, beta[start:stop, i_anc, :])

    # scale variance of pheno_g to hsq
    std_scale = np.sqrt(hsq / np.var(pheno_g, axis=0))
    pheno_g *= std_scale
    beta *= std_scale

    pheno_e = np.zeros(pheno_g.shape)
    for i_sim in range(n_sim):
        pheno_e[:, i_sim] = np.random.normal(
            loc=0.0, scale=np.sqrt(1 - hsq), size=n_indiv
        )

    pheno = pheno_g + pheno_e

    # if `cov_cols` are specified, add the covariates to the phenotype
    if cov_cols is not None:
        # if `cov_effects` are not set, set to random normal values
        if cov_effects is None:
            cov_effects = np.random.normal(size=len(cov_cols))
        # add the covariates to the phenotype
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset.indiv[cov_col].values
        pheno += np.dot(cov_values, cov_effects).reshape((n_indiv, 1))

    return {
        "beta": beta,
        "pheno_g": pheno_g,
        "pheno": pheno,
    }


# TODO: check https://github.com/TalShor/SciLMM/blob/master/scilmm/Estimation/HE.py


def quant_pheno_grm(
    dset: xr.Dataset,
    grm: Union[str, List[str], dict],
    var: Dict[str, float],
    cov_cols: List[str] = None,
    cov_effects: List[float] = None,
    n_sim=10,
):
    """Simulate continuous phenotype of admixed individuals [continuous] using GRM
    grm: str, list of str or dict
        column name(s) of GRM, or a dict of {name: grm}
        Don't include the identity matrix, the indentify matrix representing environemntal
        factor will be added to the list automatically
    var: dict of {str: float}
        dictionary of variance explained by the GRM effect, use 'e' to set the variance of
        environmental effectse.g. {'K1': 0.5, 'K2': 0.5, 'e': 1.0}
    gamma: float, optional
        Correlation between the genetic effects from two ancestral backgrounds, by default None
    cov_cols: List[str], optional
        list of covariates to include as covariates, by default None
    cov_effects: List[float], optional
        list of the effect of each covariate, by default None
        for each simulation, the cov_effects will be the same
    n_sim : int, optional
        number of simulations, by default 10

    Returns
    -------
    A dictionary containing the following variables:
        - pheno: (n_indiv, n_sim) simulated phenotype
        - cov_effects: (n_cov,) simulated covariate effects
    """
    # get the dictionary of all the GRMs (including environmental)
    # name -> (grm, var)
    if isinstance(grm, dict):
        # obtain from the list
        grm_var = {k: (grm[k], var[k]) for k in grm}
    elif isinstance(grm, list):
        grm_var = {k: (dset[k].data, var[k]) for k in grm}
    elif isinstance(grm, str):
        grm_var = {grm: (dset[grm].data, var[grm])}
    else:
        raise ValueError("`grm` must be a dictionary, list or string")

    # add environemntal effect
    n_indiv = dset.dims["indiv"]
    grm_var["e"] = (np.eye(n_indiv), var["e"])

    covariance = da.zeros((n_indiv, n_indiv))
    for name in grm_var:
        covariance += grm_var[name][0] * grm_var[name][1]

    # fixed effects
    # if `cov_cols` are specified, add the covariates to the phenotype
    if cov_cols is not None:
        # if `cov_effects` are not set, set to random normal values
        if cov_effects is None:
            cov_effects = np.random.normal(size=len(cov_cols))
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col + "@indiv"].values
        fixed_effects = np.dot(cov_values, cov_effects)
    else:
        fixed_effects = np.zeros(n_indiv)
    # simulate phenotype
    ys = np.random.multivariate_normal(
        mean=fixed_effects,
        cov=covariance,
        size=n_sim,
    ).T

    return {"pheno": ys, "cov_effects": cov_effects}


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


def binary_pheno(
    dset: admix.Dataset,
    hsq: float = 0.5,
    cor: float = 1,
    n_causal: int = None,
    case_prevalence: float = 0.1,
    beta: np.ndarray = None,
    cov_cols: List[str] = None,
    cov_effects: List[float] = None,
    n_sim=10,
    method="probit",
):
    """
    Simulate under liability threshold. First simulate quantative traits in the liability
    scale. With the given `case_prevalence`, convert liability to binary traits.

    Parameters
    ----------
    hsq: if method is "probit", this corresponds to the proportion of variance explained by the genotype
        in the liability scale.

        if method is "logit", this corresponds to the variance explained of (X\beta),
        and hsq can be larger than 1, in the case.
    """

    from scipy import stats

    assert method in ["probit", "logit"]

    # build a skeleton of phenotype using the code from quant_pheno

    if method == "probit":
        quant_sim = quant_pheno(
            dset=dset,
            hsq=hsq,
            cor=cor,
            n_causal=n_causal,
            beta=beta,
            cov_cols=cov_cols,
            cov_effects=cov_effects,
            n_sim=n_sim,
        )

        beta = quant_sim["beta"]
        pheno_g = quant_sim["pheno_g"]
        qpheno = quant_sim["pheno"]

        case_threshold = -stats.norm.ppf(case_prevalence)

        pheno = np.zeros_like(qpheno)
        pheno[qpheno < case_threshold] = 0
        pheno[qpheno >= case_threshold] = 1

        return {
            "beta": beta,
            "pheno_g": pheno_g,
            "pheno": pheno.astype(int),
        }

    elif method == "logit":
        from scipy.optimize import fsolve
        from scipy.special import logit, expit

        quant_sim = quant_pheno(
            dset=dset,
            hsq=0.5,
            cor=cor,
            n_causal=n_causal,
            beta=beta,
            cov_cols=cov_cols,
            cov_effects=cov_effects,
            n_sim=n_sim,
        )
        beta = quant_sim["beta"]
        pheno_g = quant_sim["pheno_g"]

        # scale variance of pheno_g to hsq
        std_scale = np.sqrt(hsq / np.var(pheno_g, axis=0))
        pheno_g *= std_scale
        beta *= std_scale
        pheno = np.zeros_like(pheno_g)
        for sim_i in range(n_sim):
            # find an intercept, such that the expectation is case_prevalence.
            func = lambda b: np.mean(expit(b + pheno_g[:, sim_i])) - case_prevalence
            intercept = fsolve(func, logit(case_prevalence))
            pheno[:, sim_i] = np.random.binomial(
                1, expit(intercept + pheno_g[:, sim_i])
            )
        return {
            "beta": beta,
            "pheno_g": pheno_g,
            "pheno": pheno.astype(int),
        }
    else:
        raise NotImplementedError


def quant_pheno_1pop(
    dset: xr.Dataset,
    var_g: float = None,
    var_e: float = None,
    n_causal: int = None,
    beta: np.ndarray = None,
    cov_cols: List[str] = None,
    cov_effects: List[float] = None,
    n_sim=10,
) -> dict:
    """Simulate continuous phenotype for a single population [continuous]

    Parameters
    ----------
    dset: xr.Dataset
        Dataset with the following variables:
            - `geno`: (n_indiv, n_snp) genotype of each individual
    var_g: float or np.ndarray
        Variance explained by the genotype effect
    var_e: float
        Variance explained by the effect of the environment
    n_causal: int, optional
        number of causal variables, by default None
    beta: np.ndarray, optional
        Effect sizes
    cov_cols: List[str], optional
        list of covariates to include as covariates, by default None
    cov_effects: List[float], optional
        list of the effect of each covariate, by default None
        for each simulation, the cov_effects will be the same
    n_sim : int, optional
        number of simulations, by default 10


    Returns
    -------
    beta: np.ndarray
        simulated effect sizes (2 * n_snp, n_sim)
    phe_g: np.ndarray
        simulated genetic component of phenotypes (n_indiv, n_sim)
    phe: np.ndarray
        simulated phenotype (n_indiv, n_sim)
    """

    n_indiv, n_snp = dset.dims["indiv"], dset.dims["snp"]
    geno = dset.geno.data
    centered_geno = geno - da.nanmean(geno, axis=0)
    # snp_var = da.nanvar(centered_geno, axis=0).compute()

    # simulate effect sizes
    if beta is None:
        if n_causal is None:
            # n_causal = n_snp if `n_causal` is not specified
            n_causal = n_snp
        assert (
            var_g is not None and var_e is not None
        ), "`var_g` and `var_e` must be specified"
        # if `beta` is not specified, simulate effect sizes
        beta = np.zeros((n_snp, n_sim))
        for i_sim in range(n_sim):
            cau = sorted(
                np.random.choice(np.arange(n_snp), size=n_causal, replace=False)
            )
            i_beta = np.random.normal(
                loc=0.0,
                scale=np.sqrt(var_g / n_causal),
                size=n_causal,
            )
            beta[cau, i_sim] = i_beta
    else:
        assert (var_g is None) and (
            n_causal is None
        ), "If `beta` is specified, `var_g`, `var_e`, and `n_causal` must be specified"
        assert beta.shape == (n_snp,) or beta.shape == (
            n_snp,
            n_sim,
        ), "`beta` must be of shape (n_snp,) or (n_snp, n_sim)"
        if beta.shape == (n_snp,):
            # replicate `beta` for each simulation
            beta = np.repeat(beta[:, np.newaxis], n_sim, axis=2)

    pheno_g = admix.data.geno_mult_mat(centered_geno, beta)

    # standardize the phe_g so that the variance of g is `var_g`
    std_scale = np.sqrt(var_g / np.var(pheno_g, axis=0))
    pheno_g *= std_scale
    beta *= std_scale

    pheno_e = np.zeros(pheno_g.shape)
    for i_sim in range(n_sim):
        pheno_e[:, i_sim] = np.random.normal(
            loc=0.0, scale=np.sqrt(var_e), size=n_indiv
        )

    pheno = pheno_g + pheno_e
    pheno_g, pheno = dask.compute((pheno_g, pheno))[0]
    # if `cov_cols` are specified, add the covariates to the phenotype
    if cov_cols is not None:
        # if `cov_effects` are not set, set to random normal values
        if cov_effects is None:
            cov_effects = np.random.normal(size=len(cov_cols))
        # add the covariates to the phenotype
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col].values
        pheno += np.dot(cov_values, cov_effects).reshape((n_indiv, 1))

    return {
        "beta": pd.DataFrame(
            beta, index=dset.snp.values, columns=[f"SIM_{i}" for i in range(n_sim)]
        ),
        "pheno_g": pd.DataFrame(
            pheno_g, index=dset.indiv.values, columns=[f"SIM_{i}" for i in range(n_sim)]
        ),
        "pheno": pd.DataFrame(
            pheno, index=dset.indiv.values, columns=[f"SIM_{i}" for i in range(n_sim)]
        ),
        "cov_effects": cov_effects,
    }

import numpy as np
from scipy import linalg
import pandas as pd
import xarray as xr
from typing import List


def trace_mul(a, b):
    """
    Trace of two matrix inner product
    """
    assert np.all(a.shape == b.shape)
    return np.sum(a.flatten() * b.flatten())


def gen_cor(
    dset: xr.Dataset,
    admix_grm: dict,
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
    method: str = "HE",
    estimand: str = "var",
):
    """Estimate genetic correlation given a dataset, admixture GRM, phenotypes, and covariates.

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to estimate correlation from.
    admix_grm: dict
        Admixture GRMs
    pheno: np.ndarray
        Phenotypes to estimate genetic correlation. If a matrix is provided, then each
        column is treated as a separate phenotype.
    cov_cols: list, optional
        List of covariate columns.
    cov_intercept: bool, optional
        Whether to include intercept in covariate matrix.
    method: str, optional
        Method to use for estimation. Valid options are:
            - "HE" (default): Haseman–Elston (HE) regression.
            - "RHE": Randomized Haseman–Elston (RHE) regression.
            - "REML": REML regression.
    estimand: str, optional
        Estimand to use for estimation. Valid options are:
            - "var": Only genetic variance.
            - "var+gamma": Both genetic variance and covariance.
            - "var1+var2+gamma": Genetic variances from two backgrounds and covariance.
    """
    assert estimand in ["var", "var+gamma", "var1+var2+gamma"]
    assert method in ["HE", "RHE", "REML"]
    n_indiv = dset.dims["indiv"]
    K1, K2, K12 = [admix_grm[k] for k in ["K1", "K2", "K12"]]

    # build the covariate matrix
    if cov_cols is not None:
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col + "@indiv"].values
        if cov_intercept:
            cov_values = np.c_[np.ones((n_indiv, 1)), cov_values]
    else:
        cov_values = None
        if cov_intercept:
            cov_values = np.ones((n_indiv, 1))
    # build projection matrix from covariate matrix
    if cov_values is None:
        cov_proj_mat = np.eye(n_indiv)
    else:
        cov_proj_mat = np.eye(n_indiv) - np.linalg.multi_dot(
            [cov_values, np.linalg.inv(np.dot(cov_values.T, cov_values)), cov_values.T]
        )

    if pheno.ndim == 1:
        pheno = pheno.reshape((-1, 1))
    assert pheno.shape[0] == n_indiv

    n_pheno = pheno.shape[1]

    pheno = np.dot(cov_proj_mat, pheno)
    quad_form_func = lambda x, A: np.dot(np.dot(x.T, A), x)
    if method == "HE":
        df_rls = []
        if estimand == "var":
            A = K1 + K2 + K12 + K12.T
            A = np.dot(A, cov_proj_mat)
            HE_design = np.array(
                [
                    [trace_mul(A, A), np.trace(A)],
                    [np.trace(A), np.linalg.matrix_rank(cov_proj_mat)],
                ]
            )
            for i_pheno in range(n_pheno):
                # build response vector
                HE_response = np.array(
                    [
                        quad_form_func(pheno[:, i_pheno], A),
                        quad_form_func(pheno[:, i_pheno], np.eye(n_indiv)),
                    ]
                )
                rls = linalg.solve(HE_design, HE_response)
                df_rls.append(rls)
            df_rls = pd.DataFrame(np.vstack(df_rls), columns=["sigma_g", "sigma_e"])
        elif estimand == "var+gamma":
            A1 = K1 + K2
            A2 = K12 + K12.T

            A1 = np.dot(A1, cov_proj_mat)
            A2 = np.dot(A2, cov_proj_mat)

            HE_design = np.array(
                [
                    [trace_mul(A1, A1), trace_mul(A1, A2), np.trace(A1)],
                    [trace_mul(A2, A1), trace_mul(A2, A2), np.trace(A2)],
                    [np.trace(A1), np.trace(A2), np.linalg.matrix_rank(cov_proj_mat)],
                ]
            )
            for i_pheno in range(n_pheno):

                # build response vector
                HE_response = np.array(
                    [
                        quad_form_func(pheno[:, i_pheno], A1),
                        quad_form_func(pheno[:, i_pheno], A2),
                        quad_form_func(pheno[:, i_pheno], np.eye(n_indiv)),
                    ]
                )
                rls = linalg.solve(HE_design, HE_response)
                df_rls.append(rls)
            df_rls = pd.DataFrame(
                np.vstack(df_rls), columns=["sigma_g", "gamma", "sigma_v"]
            )
        elif estimand == "var1+var2+gamma":
            raise NotImplementedError

    return df_rls
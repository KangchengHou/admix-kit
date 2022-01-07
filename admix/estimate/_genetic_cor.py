import numpy as np
from scipy import linalg
import pandas as pd
import xarray as xr
from typing import List, Union
import admix
import tempfile
from ..utils import cd


def trace_mul(a, b):
    """
    Trace of two matrix inner product
    """
    assert np.all(a.shape == b.shape)
    return np.sum(a.flatten() * b.flatten())


def gen_cor(
    dset: admix.Dataset,
    grm: Union[str, List[str], dict],
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
    method: str = "HE",
):
    """Estimate genetic correlation given a dataset, admixture GRM, phenotypes, and covariates.

    Parameters
    ----------
    dset: admix.Dataset
        Dataset to estimate correlation from.
    grm: str, list of str or dict
        column name(s) of GRM, or a dict of {name: grm}
        Don't include the identity matrix, the indentify matrix representing environemntal
        factor will be added to the list automatically
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
            - "HE-gcta": HE regression as implemented in GCTA.
            - "REML-gcta": REML as implemented in GCTA.
    """
    assert method in ["HE", "RHE", "REML"]
    n_indiv = dset.dims["indiv"]
    # get the dictionary of all the GRMs (including environmental)
    # name -> grm
    if isinstance(grm, dict):
        # obtain from the list
        grm_dict = {k: grm[k] for k in grm}
    elif isinstance(grm, list):
        grm_dict = {k: dset[k].data for k in grm}
    elif isinstance(grm, str):
        grm_dict = {grm: dset[grm].data}
    else:
        raise ValueError("`grm` must be a dictionary, list or string")

    # add the environmental GRM
    n_indiv = dset.dims["indiv"]
    grm_dict["e"] = np.eye(n_indiv)

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
        grm_list = [np.dot(grm_dict[name], cov_proj_mat) for name in grm_dict]
        n_grm = len(grm_list)
        HE_design = np.zeros((n_grm, n_grm))

        # fill in `HE_design`
        for i in range(n_grm):
            for j in range(n_grm):
                if i <= j:
                    HE_design[i, j] = (grm_list[i] * grm_list[j]).sum()
                    HE_design[j, i] = HE_design[i, j]

        # build response vector for each phenotype
        for i_pheno in range(n_pheno):
            HE_response = np.zeros(n_grm)
            for i in range(n_grm):
                HE_response[i] = quad_form_func(pheno[:, i_pheno], grm_list[i])
            df_rls.append(linalg.solve(HE_design, HE_response))
        df_rls = pd.DataFrame(np.vstack(df_rls), columns=[name for name in grm_dict])

    return df_rls
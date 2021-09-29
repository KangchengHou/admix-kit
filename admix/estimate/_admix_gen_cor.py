import numpy as np
from scipy import linalg
import xarray as xr
from typing import List, Tuple
import admix


def admix_gen_cor(
    dset: xr.Dataset,
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
):
    """Estimate genetic correlation given a dataset, phenotypes, and covariates.
    This is a very specialized function that tailed for estimating the genetic correlation
    for variants in different local ancestry backgrounds.

    See details in https://www.nature.com/articles/s41467-020-17576-9#MOESM1

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to estimate correlation from.
    pheno: np.ndarray
        Phenotypes to estimate genetic correlation. If a matrix is provided, then each
        column is treated as a separate phenotype.
    cov_cols: list, optional
        List of covariate columns.
    cov_intercept: bool, optional
        Whether to include intercept in covariate matrix.
    """
    n_indiv = dset.dims["indiv"]

    # build the covariate matrix
    if cov_cols is not None:
        cov_values = np.zeros((n_indiv, len(cov_cols)))
        for i_cov, cov_col in enumerate(cov_cols):
            cov_values[:, i_cov] = dset[cov_col].values
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
    n_indiv = dset.dims["indiv"]
    n_snp = dset.dims["snp"]
    # apa = admix.tools.allele_per_anc(dset, center=True, inplace=False).astype(float)
    # a1, a2 = apa[:, :, 0], apa[:, :, 1]
    # grm_list = [
    #     (np.dot(a1, a1.T) + np.dot(a2, a2.T)) / n_snp,
    #     (np.dot(a1, a2.T) + np.dot(a1, a2.T).T) / n_snp,
    #     np.eye(n_indiv),
    # ]

    grm_list = [dset["A1"].data, dset["A2"].data, np.eye(n_indiv)]
    grm_list = [np.dot(grm, cov_proj_mat) for grm in grm_list]

    # multiply cov_proj_mat
    n_grm = len(grm_list)
    design = np.zeros((n_grm, n_grm))
    for i in range(n_grm):
        for j in range(n_grm):
            if i <= j:
                design[i, j] = (grm_list[i] * grm_list[j]).sum()
                design[j, i] = design[i, j]

    rls_list: List[Tuple] = []
    for i_pheno in range(n_pheno):
        response = np.zeros(n_grm)
        for i in range(n_grm):
            response[i] = quad_form_func(pheno[:, i_pheno], grm_list[i])

        # point estimate
        var_comp = linalg.solve(
            design,
            response,
        )

        # variance-covariance matrix
        inv_design = linalg.inv(design)
        Sigma = np.zeros_like(grm_list[0])
        for i in range(n_grm):
            Sigma += var_comp[i] * grm_list[i]
        Sigma_grm_list = [np.dot(Sigma, grm) for grm in grm_list]

        var_response = np.zeros((n_grm, n_grm))
        for i in range(n_grm):
            for j in range(n_grm):
                if i <= j:
                    var_response[i, j] = (
                        2 * (Sigma_grm_list[i] * Sigma_grm_list[j]).sum()
                    )
                    var_response[j, i] = var_response[i, j]
        var_comp_var = np.linalg.multi_dot([inv_design, var_response, inv_design])
        rls_list.append((var_comp, var_comp_var))
    return rls_list
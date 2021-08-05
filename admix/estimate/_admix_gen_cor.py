import numpy as np
from scipy import linalg
import pandas as pd
import xarray as xr
from typing import List, Union
import admix
import tempfile
from .._utils import cd
import dask
from tqdm import tqdm
import dask.array as da


def admix_gen_cor(
    dset: xr.Dataset,
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
    n_jackknife_blocks: int = 5,
):
    """Estimate genetic correlation given a dataset, phenotypes, and covariates.
    This is a very specialized function that tailed for estimating the genetic correlation
    for variants in different local ancestry backgrounds. The function is seperated from
    the general class for computational efficiency of implmentation (e.g. using streaming
    implmentation for block jackknife).

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

    # streaming block jackknife
    jackknife_snps = np.array_split(np.arange(dset.dims["snp"]), n_jackknife_blocks)
    jackknife_n_snp = [len(x) for x in jackknife_snps]
    total_n_snp = sum(jackknife_n_snp)
    jackknife_design: List[np.ndarray] = []
    jackknife_response: List[np.ndarray] = []

    apa = admix.tools.allele_per_anc(dset, center=True, inplace=False).astype(float)

    for i_block in tqdm(range(n_jackknife_blocks)):
        # subset snps
        block_snps = jackknife_snps[i_block]
        a1, a2 = apa[:, block_snps, 0], apa[:, block_snps, 1]
        grm_list = [
            np.dot(a1, a1.T) + np.dot(a2, a2.T),
            np.dot(a1, a2.T) + np.dot(a1, a2.T).T,
            np.eye(n_indiv),
        ]

        # multiply cov_proj_mat
        n_grm = len(grm_list)
        design = np.zeros((n_grm, n_grm))
        for i in range(n_grm):
            for j in range(n_grm):
                if i <= j:
                    design[i, j] = (grm_list[i] * grm_list[j]).sum()

        jackknife_design.append(design)

        # cope with one phenotype for now
        # TODO: add multiple phenotypes
        response = np.zeros(n_grm)
        for i in range(n_grm):
            response[i] = quad_form_func(pheno, grm_list[i])
        jackknife_response.append(response)

    # perform leave-block-out jackknife regression

    total_design = np.sum(jackknife_design, axis=0)
    total_response = np.sum(jackknife_response, axis=0)

    df_rls = [
        linalg.solve(
            (total_design - jackknife_design[i_block])
            / np.square(total_n_snp - jackknife_n_snp[i_block]),
            (total_response - jackknife_response[i_block])
            / (total_n_snp - jackknife_n_snp[i_block]),
        )
        for i_block in range(n_jackknife_blocks)
    ]
    df_rls = pd.DataFrame(df_rls, index=np.arange(n_jackknife_blocks))
    print(df_rls)
    print(
        linalg.solve(
            total_design / np.square(total_n_snp), total_response / total_n_snp
        )
    )

    return df_rls
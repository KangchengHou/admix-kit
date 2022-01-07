import numpy as np
from scipy import linalg
import pandas as pd
import xarray as xr
from typing import List, Union
import admix
import tempfile
from ..utils import cd
import dask


def variance_component(
    dset: admix.Dataset,
    grm: Union[str, List[str], dict],
    pheno: np.ndarray,
    cov_cols: List[str] = None,
    cov_intercept: bool = True,
    method: str = "HE",
):
    """
    Common interface for fitting variance components.
    Including HE-regression and REML.

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

    assert method in ["HE", "HE-gcta", "RHE", "REML"]
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

    if pheno.ndim == 1:
        pheno = pheno.reshape((-1, 1))
    assert pheno.shape[0] == n_indiv

    n_pheno = pheno.shape[1]

    if method == "HE" or method == "HE-gcta":

        rls_list = HE_reg(
            grm_list=[grm_dict[name] for name in grm_dict],
            pheno=pheno,
            cov=cov_values,
            method="built-in" if method == "HE" else "gcta",
        )
        if method == "HE":
            columns = [name for name in grm_dict] + ["e"]
        elif method == "HE-gcta":
            columns = [name + "/total" for name in grm_dict]
        else:
            raise ValueError("Invalid method")
        df_rls = pd.DataFrame(rls_list, columns=columns)

    elif method == "REML":
        rls_list = REML(
            grm_list=[grm_dict[name] for name in grm_dict], pheno=pheno, cov=cov_values
        )
    else:
        raise NotImplementedError("{} is not implemented".format(method))

    return df_rls


def HE_reg(
    grm_list: List,
    pheno: np.ndarray,
    cov: np.ndarray = None,
    method="built-in",
):
    """
    Estimate the variance components for the given phenotype with the
    Haseman-Elston (HE) regression. Note that the enviornmental GRM (identity matrix)
    should be not included in the list of GRMs.

    Parameters
    ----------
    grm_list : list of array-like
        List of GRMs to be used.
    pheno : array-like
        Phenotypes to estimate genetic correlation. If a matrix is provided, then each
        column is treated as a separate phenotype.
    cov : array-like, optional
        Covariates (n_indiv, n_cov) to be used in the regression. If None, then no covariates are used.
    method : str, optional
        Method to use.
            "built-in": built-in implementation in admix.estimators.HE_reg
            "gcta": use GCTA to estimate the variance components
    Returns
    -------
    df_rls : pandas.DataFrame
        DataFrame with the regression coefficients.

    """

    quad_form = lambda x, A: np.dot(np.dot(x.T, A), x)

    n_indiv = grm_list[0].shape[0]
    grm_list = grm_list + [np.eye(n_indiv)]
    n_grm = len(grm_list)
    n_pheno = pheno.shape[1]

    # list of results (one for each simulation)
    rls_list = []
    assert method in [
        "built-in",
        "gcta",
    ], "`method` must be one of: ['built-in', 'gcta']"

    # build projection matrix from covariate matrix
    if cov is None:
        cov_proj_mat = np.eye(n_indiv)
    else:
        # TODO: check colinearity of the covariates?
        cov_proj_mat = np.eye(n_indiv) - np.linalg.multi_dot(
            [
                cov,
                np.linalg.inv(np.dot(cov.T, cov)),
                cov.T,
            ]
        )
    # project the grm and the phenotypes to the new space
    grm_list = [np.dot(grm, cov_proj_mat) for grm in grm_list]
    pheno = np.dot(cov_proj_mat, pheno)

    if method == "built-in":
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
                HE_response[i] = quad_form(pheno[:, i_pheno], grm_list[i])
            rls_list.append(linalg.solve(HE_design, HE_response))

    elif method == "gcta":
        # creating dummy individual IDs
        df_id = pd.DataFrame(
            {
                "FID": np.arange(n_indiv).astype(str),
                "IID": np.arange(n_indiv).astype(str),
            }
        )

        # create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(tmp_dir)
            with cd(tmp_dir):
                f = open("mgrm.txt", "w")
                # the last grm corresponds to identity matrix, which will be coped
                # internally by GCTA
                for i_grm, grm in enumerate(grm_list[:-1]):
                    name = f"grm_{i_grm}"
                    # NOTE: the degree of freedom for GCTA implementation is not
                    # properly set up. Should not be a big problem for the current.

                    # NOTE: n_snps will not be used by GCTA in this HE regression.
                    # so we just set it to some arbitrary value `1`.
                    admix.io.write_gcta_grm(
                        file_prefix=name,
                        grm=dask.compute(np.dot(grm, cov_proj_mat))[0],
                        df_id=df_id,
                        n_snps=np.array([1] * n_indiv),
                    )
                    f.write(f"{name}\n")
                f.close()
                df_pheno = df_id.copy()
                for i_pheno in range(n_pheno):
                    df_pheno["PHENO"] = np.dot(cov_proj_mat, pheno[:, i_pheno])
                    df_pheno.to_csv("pheno.txt", sep="\t", index=False, header=False)

                    # run GCTA
                    admix.tools.gcta(
                        "--HEreg --mgrm mgrm.txt --pheno pheno.txt --out result"
                    )
                    df_rls = pd.read_csv(
                        "result.HEreg",
                        delim_whitespace=True,
                        skiprows=1,
                        nrows=n_grm,
                    )
                    rls_list.append(df_rls["Estimate"][1:].values)

    else:
        raise NotImplementedError

    return rls_list


def REML(grm_list: List, pheno: np.ndarray, cov: np.ndarray):
    """
    Estimate the variance components for the given phenotype with REML.
    Currently only a GCTA wrapper is provided.
    Parameters
    ----------
    grm_list : list of array-like
        List of GRMs to be used.
    pheno : array-like
        Phenotypes to estimate genetic correlation. If a matrix is provided, then each
        column is treated as a separate phenotype.
    cov : array-like
        Covariates to be used. (n_indiv, n_cov)
    """
    # creating dummy individual IDs
    n_indiv = grm_list[0].shape[0]
    n_pheno = pheno.shape[1]
    n_grm = len(grm_list)

    df_id = pd.DataFrame(
        {
            "FID": np.arange(n_indiv).astype(str),
            "IID": np.arange(n_indiv).astype(str),
        }
    )

    rls_list = []

    # create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        with cd(tmp_dir):
            f = open("mgrm.txt", "w")
            # the last grm corresponds to identity matrix, which will be coped
            # internally by GCTA, so we don't need to write it
            for i_grm, grm in enumerate(grm_list[:-1]):
                name = f"grm_{i_grm}"
                # NOTE: the degree of freedom for GCTA implementation is not
                # properly set up. Should not be a big problem for the current.

                # NOTE: n_snps will not be used by GCTA in this HE regression.
                # so we just set it to some arbitrary value `1`.
                admix.io.write_gcta_grm(
                    file_prefix=name,
                    grm=grm.compute(),
                    df_id=df_id,
                    n_snps=np.array([1] * n_indiv),
                )
                f.write(f"{name}\n")
            f.close()
            df_pheno = df_id.copy()
            df_covar = df_id.copy()
            df_covar[[f"COV{cov_i}" for cov_i in range(cov.shape[1])]] = cov
            df_covar.to_csv("covar.txt", sep="\t", index=False, header=False)

            # write covarites
            for i_pheno in range(n_pheno):
                df_pheno["PHENO"] = pheno[:, i_pheno]
                df_pheno.to_csv("pheno.txt", sep="\t", index=False, header=False)

                # run GCTA
                admix.tools.gcta(
                    "--reml --mgrm mgrm.txt --pheno pheno.txt --qcovar covar.txt --reml-est-fix --reml-no-constrain --out result"
                )

                df_rls = pd.read_csv(
                    "result.hsq",
                    delim_whitespace=True,
                    nrows=n_grm,
                )
                rls_list.append(df_rls)

    return rls_list
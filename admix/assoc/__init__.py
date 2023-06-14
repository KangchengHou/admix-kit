import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import admix
from typing import Any, Dict, List, Optional
import dask.array as da
import admix
from statsmodels.tools import sm_exceptions
import warnings
import itertools

warnings.filterwarnings(action="error", category=sm_exceptions.ValueWarning)

__all__ = ["marginal"]


def _format_block_test(
    var: np.ndarray,
    cov: np.ndarray,
    pheno: np.ndarray,
    var_size: int,
    test_vars: np.ndarray,
):
    n_indiv = var.shape[0]
    assert (
        cov.shape[0] == n_indiv
    ), "Number of individuals in genotype and covariate do not match"
    assert pheno.ndim == 1, "Phenotype must be a vector"
    assert (
        pheno.shape[0] == n_indiv
    ), "Number of individuals in genotype and phenotype do not match"
    assert var_size > 0, "Variable size must be greater than 0"
    assert (
        var.shape[1] % var_size == 0
    ), f"Number of variables in var ({var.shape[1]}) must be a multiple of var_size ({var_size})"
    n_var = var.shape[1] // var_size

    if isinstance(test_vars, List):
        test_vars = np.array(test_vars)

    assert np.all(test_vars < var_size), "test_vars must be less than var_size"

    # fill covariates
    n_cov = cov.shape[1]

    if isinstance(var, da.Array):
        var = var.compute()

    assert isinstance(var, np.ndarray), "var must be a numpy array"
    return var, n_var, n_cov, test_vars


def _block_het_test(
    var: np.ndarray,
    cov: np.ndarray,
    pheno: np.ndarray,
    var_size: int,
    test_vars: np.ndarray,
    fast: bool,
    family: str,
    logistic_kwargs: Dict[str, Any] = dict(),
):
    admix.logger.info(
        "Currently HET test is implemented through statsmodels, which can be slow. "
        "Pass in small amount of data whenever possible."
    )

    ## Format data
    var, n_var, n_cov, test_vars = _format_block_test(
        var, cov, pheno, var_size, test_vars
    )

    # format reduced variable matrix
    n_indiv = var.shape[0]
    reduced_var_size = var_size - len(test_vars) + 1
    design_full = np.zeros((n_indiv, var_size + n_cov))
    design_full[:, var_size : var_size + n_cov] = cov
    design_reduced = np.zeros((n_indiv, reduced_var_size + n_cov))
    design_reduced[:, reduced_var_size : reduced_var_size + n_cov] = cov
    other_vars = np.array([i for i in range(var_size) if i not in test_vars])
    shared_param_index = np.concatenate(
        [other_vars, np.arange(var_size, var_size + n_cov)]
    ).astype(int)

    # statsmodels implementation
    if family == "linear":
        reg_method = lambda pheno, design, start_params=None: sm.OLS(
            pheno, design, missing="drop"
        ).fit(disp=0, start_params=start_params)

    elif family == "logistic":
        reg_method = lambda pheno, design, start_params=None: sm.Logit(
            pheno, design, missing="drop"
        ).fit(disp=0, start_params=start_params)
    else:
        raise NotImplementedError

    # legacy implementation of using F-test
    # ftest_mat = np.zeros([len(test_vars) - 1, design_full.shape[1]])
    # for i in range(len(test_vars) - 1):
    #     ftest_mat[i, test_vars[i]] = 1
    #     ftest_mat[i, test_vars[i + 1]] = -1

    res = np.zeros((n_var, var_size * 2 + 2))

    for i in range(n_var):
        design_full[:, 0:var_size] = var[:, i * var_size : (i + 1) * var_size]
        model_full = reg_method(pheno, design_full)

        # coefficients
        res[i, 0 : var_size * 2 : 2] = model_full.params[0:var_size]
        # standard errors
        res[i, 1 : var_size * 2 : 2] = model_full.bse[0:var_size]
        res[i, -2] = model_full.nobs

        design_reduced[:, 0] = var[:, test_vars + i * var_size].sum(axis=1)
        if len(other_vars) > 0:
            design_reduced[:, 1:reduced_var_size] = var[:, other_vars + i * var_size]

        model_reduced = reg_method(
            pheno,
            design_reduced,
            start_params=np.concatenate([[0.0], model_full.params[shared_param_index]]),
        )

        if family == "linear" or family == "logistic":
            p = stats.chi2.sf(
                -2 * (model_reduced.llf - model_full.llf),
                (model_full.df_model - model_reduced.df_model),
            )
            res[i, -1] = p
        else:
            raise NotImplementedError
    return res


def _block_test(
    var: np.ndarray,
    cov: np.ndarray,
    pheno: np.ndarray,
    var_size: int,
    test_vars: np.ndarray,
    fast: bool,
    family: str,
    logistic_kwargs: Dict[str, Any] = dict(),
) -> np.ndarray:
    """
    Perform association testing for a block of variables

    Parameters
    ----------
    var : np.ndarray
        (n_indiv, n_var x var_size) variable matrix
    cov : np.ndarray
        (n_indiv, n_cov) covariate matrix
    pheno : np.ndarray
        (n_snp) phenotype matrix
    var_size : int
        Number of variables for each test
    test_vars : List[int]
        Index of variables to test

    Returns
    -------
    np.ndarray
        (n_snp, n_test_var * 2 + 2) association testing information.
        - The first n_test_var * 2 columns are the effect size (odd columns)
        and standard error (even columns) for each variable.
        - The last two columns are the number of individuals and p-value.
    """
    var, n_var, n_cov, test_vars = _format_block_test(
        var, cov, pheno, var_size, test_vars
    )

    if fast:
        try:
            import tinygwas
        except ImportError:
            raise ImportError(
                "\nplease install tinygwas:\n\n"
                "\tpip install git+https://github.com/bogdanlab/tinygwas.git#egg=tinygwas"
            )

        res = np.zeros((n_var, var_size * 2 + 2))
        if family == "linear":
            tinygwas.linear_f_test(var, cov, pheno, var_size, test_vars, res)
            f_stats = res[:, -1]
            n_indiv_test = res[:, -2]

            res[:, -1] = stats.f.sf(
                f_stats, len(test_vars), n_indiv_test - n_cov - var_size
            )

        elif family == "logistic":
            if "max_iter" not in logistic_kwargs:
                logistic_kwargs["max_iter"] = 100
            if "tol" not in logistic_kwargs:
                logistic_kwargs["tol"] = 1e-6

            tinygwas.logistic_lrt(
                var,
                cov,
                pheno,
                var_size,
                test_vars,
                res,
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            # convert lrt diff to pvalue
            res[:, -1] = stats.chi2.sf(2 * res[:, -1], len(test_vars))
        else:
            raise ValueError(f"Unknown family: {family}")
    else:
        # statsmodels implementation
        n_indiv = var.shape[0]
        design = np.zeros((n_indiv, var_size + n_cov))
        design[:, var_size : var_size + n_cov] = cov

        if family == "linear":
            reg_method = lambda pheno, design, start_params=None: sm.OLS(
                pheno, design, missing="drop"
            ).fit(disp=0, start_params=start_params)
        elif family == "logistic":
            reg_method = lambda pheno, design, start_params=None: sm.Logit(
                pheno, design, missing="drop"
            ).fit(disp=0, start_params=start_params)
        else:
            raise NotImplementedError

        reduced_index = np.concatenate(
            [
                [i for i in range(var_size) if i not in test_vars],
                np.arange(var_size, var_size + n_cov),
            ]
        ).astype(int)

        f_test_r_matrix = np.zeros((len(test_vars), design.shape[1]))
        for i, v in enumerate(test_vars):
            f_test_r_matrix[i, v] = 1

        res = np.zeros((n_var, var_size * 2 + 2))

        for i in range(n_var):
            design[:, 0:var_size] = var[:, i * var_size : (i + 1) * var_size]
            model = reg_method(pheno, design)
            # coefficients
            res[i, 0 : var_size * 2 : 2] = model.params[0:var_size]
            # standard errors
            res[i, 1 : var_size * 2 : 2] = model.bse[0:var_size]
            res[i, -2] = model.nobs
            # sample size
            if family == "linear":
                # f-test using statsmodels
                try:
                    p = model.f_test(f_test_r_matrix).pvalue.item()
                except sm_exceptions.ValueWarning:
                    p = np.nan
                res[i, -1] = p

            elif family == "logistic":
                # more than one test variables
                model_reduced = reg_method(
                    pheno,
                    design[:, reduced_index],
                    start_params=model.params[reduced_index],
                )
                # determine p-values using difference in log-likelihood
                # and difference in degrees of freedom
                p = stats.chi2.sf(
                    -2 * (model_reduced.llf - model.llf),
                    (model.df_model - model_reduced.df_model),
                )
                res[i, -1] = p
    return res


def marginal(
    dset: admix.Dataset = None,
    geno: da.Array = None,
    lanc: da.Array = None,
    pheno: np.ndarray = None,
    n_anc: Optional[int] = None,
    cov: Optional[np.ndarray] = None,
    method: str = "ATT",
    family: str = "linear",
    fast: bool = True,
):
    """Marginal association testing for one SNP at a time

    Parameters
    ----------
    dset : admix.Dataset
        data set
    geno : da.Array
        genotype (n_snp, n_indiv, 2) matrix
    lanc : da.Array
        local ancestry (n_snp, n_indiv, 2) matrix
    pheno : np.ndarray
        phenotype (n_snp, )
    n_anc : int
        number of ancestral populations, if not specified, inferred from lanc
    cov : np.ndarray, optional
        Covariate matrix, by default None. Do NOT include `1` intercept
    method : str, optional
        methods used for association testing, by default "ATT", one of
        ["ATT", "TRACTOR", "JOINT", "ADM", "SNP1", "ASE", "HET"]
    family : str, optional
        family of phenotype, by default "linear"
    fast : bool, optional
        use fast implementation, by default True

    Returns
    -------
    np.ndarray
        Association p-values for each SNP being tested

    """

    assert family in ["linear", "logistic"], "Unknown family"

    # check phenotype
    # nan values are not allowed
    assert pheno is not None, "Must specify `pheno`"
    assert np.all(np.isfinite(pheno)), "pheno must not contain NaN values"
    if family == "logistic":
        assert np.all(
            (pheno == 0) | (pheno == 1)
        ), "When family='logistic', pheno must be 0 or 1"

    if family == "logistic":
        if len(pheno) < 1000:
            admix.logger.warn(
                "logistic family is known to be unstable with small sample size (N < 1,000)"
                + "NaN values in the returned p-values may be caused by this."
            )

    # format data
    assert method in ["ATT", "TRACTOR", "JOINT", "ADM", "SNP1", "ASE", "HET"]
    if dset is not None:
        assert (geno is None) and (
            lanc is None
        ), "Cannot specify both `dset` and `geno`, `lanc`"
        geno = dset.geno
        lanc = dset.lanc
        n_anc = dset.n_anc
    else:
        assert (
            (geno is not None) and (lanc is not None) and (n_anc is not None)
        ), "Must specify `dset` or (`geno`, `lanc`, `n_anc`)"
        # convert geno and lanc to da.Array when necessary
        if not isinstance(geno, da.Array):
            geno = da.from_array(geno, chunks=-1)
        if not isinstance(lanc, da.Array):
            lanc = da.from_array(lanc, chunks=-1)

    assert np.all(geno.shape == lanc.shape), "geno and lanc must have same shape"
    n_snp, n_indiv = geno.shape[0:2]

    # process covariates
    if cov is None:
        cov = np.ones((n_indiv, 1))
    else:
        assert cov.shape[0] == n_indiv, "cov must have same number of rows as pheno"
        # prepend a column of ones to the covariates
        cov = np.hstack((np.ones((n_indiv, 1)), cov))

    assert cov is not None

    # impute missing values when needed
    if np.isnan(cov).any():
        admix.logger.info(
            "NaN found in covariates, impute with column mean for each covariate."
        )
        # fill nan with column mean
        debug_old_mean = np.nanmean(cov, axis=0)
        cov = admix.data.impute_with_mean(cov, axis=0)
        assert np.allclose(
            np.nanmean(cov, axis=0), debug_old_mean, equal_nan=True
        ), "NaN imputation failed"

    # check covariates must be full rank
    assert np.linalg.matrix_rank(cov) == cov.shape[1], "Covariates must be of full rank"

    if method == "ATT":
        # test genotype dosage
        var = geno.sum(axis=2).swapaxes(0, 1)
        var_size = 1
        var_names = ["G"]
        test_vars = [0]

    elif method == "SNP1":
        # test genotype dosage, condition on local ancestry
        var = da.empty((n_indiv, n_snp * n_anc))
        var[:, 0::n_anc] = geno.sum(axis=2).swapaxes(0, 1)
        for i in range(n_anc - 1):
            var[:, (1 + i) :: n_anc] = (lanc == i).sum(axis=2).swapaxes(0, 1)
        var_size = n_anc
        var_names = ["G"] + [f"L{i + 1}" for i in range(n_anc - 1)]
        test_vars = [0]

    elif method == "TRACTOR":
        # test ancestry-specfic genotype dosage, condition on local ancestry
        allele_per_anc = admix.data.allele_per_anc(
            geno,
            lanc,
            n_anc=n_anc,
        ).swapaxes(0, 1)
        var = da.empty((n_indiv, n_snp * (2 * n_anc - 1)))

        # allele per ancestor per SNP
        for i in range(n_anc):
            var[:, i :: (2 * n_anc - 1)] = allele_per_anc[:, :, i]

        # number of ancestries per SNP
        for i in range(n_anc - 1):
            var[:, (i + n_anc) :: (2 * n_anc - 1)] = (
                (lanc == i).sum(axis=2).swapaxes(0, 1)
            )
        var_size = 2 * n_anc - 1
        var_names = [f"G{i + 1}" for i in range(n_anc)] + [
            f"L{i + 1}" for i in range(n_anc - 1)
        ]
        test_vars = [i for i in range(n_anc)]

    elif method == "JOINT":
        # joint test of ancestry-specfic genotype dosage AND local ancestry
        allele_per_anc = admix.data.allele_per_anc(
            geno,
            lanc,
            n_anc=n_anc,
        ).swapaxes(0, 1)
        var = da.empty((n_indiv, n_snp * (2 * n_anc - 1)))

        # allele per ancestor per SNP
        for i in range(n_anc):
            var[:, i :: (2 * n_anc - 1)] = allele_per_anc[:, :, i]

        # number of ancestries per SNP
        for i in range(n_anc - 1):
            var[:, (i + n_anc) :: (2 * n_anc - 1)] = (
                (lanc == i).sum(axis=2).swapaxes(0, 1)
            )
        var_size = 2 * n_anc - 1
        var_names = [f"G{i + 1}" for i in range(n_anc)] + [
            f"L{i + 1}" for i in range(n_anc - 1)
        ]
        # test both genotype and local ancestry
        test_vars = [i for i in range(var_size)]

    elif method == "ASE":
        # test ancestry-specfic genotype dosage, without conditioning on local ancestry
        allele_per_anc = admix.data.allele_per_anc(geno, lanc, n_anc=n_anc).swapaxes(
            0, 1
        )
        var = da.empty((n_indiv, n_snp * n_anc))

        for i in range(n_anc):
            var[:, i::n_anc] = allele_per_anc[:, :, i]

        var_size = n_anc
        var_names = [f"G{i + 1}" for i in range(n_anc)]
        test_vars = [i for i in range(n_anc)]

    elif method == "ADM":
        # test local ancestry
        var = da.empty((n_indiv, n_snp * (n_anc - 1)))
        for i in range(n_anc - 1):
            var[:, i :: (n_anc - 1)] = (lanc == i).sum(axis=2).swapaxes(0, 1)
        var_size = n_anc - 1
        var_names = [f"L{i + 1}" for i in range(n_anc - 1)]
        test_vars = [i for i in range(n_anc - 1)]

    elif method == "HET":
        allele_per_anc = admix.data.allele_per_anc(geno, lanc, n_anc=n_anc).swapaxes(
            0, 1
        )
        var = da.empty((n_indiv, n_snp * n_anc))

        for i in range(n_anc):
            var[:, i::n_anc] = allele_per_anc[:, :, i]

        var_size = n_anc
        var_names = [f"G{i + 1}" for i in range(n_anc)]
        test_vars = [i for i in range(n_anc)]

    else:
        raise NotImplementedError

    # iterate over block of SNPs
    assert var.shape[1] % var_size == 0, "var must have multiple of `var_size` columns"
    assert var.shape[1] / var_size == n_snp

    res = []
    # block-by-block computation based on the chunk size of the `geno` array
    if geno is not None:
        snp_chunks = geno.chunks[0]
    else:
        assert lanc is not None
        snp_chunks = lanc.chunks[0]

    for snp_start, snp_stop in tqdm(
        admix.data.index_over_chunks(snp_chunks),
        desc="admix.assoc.marginal",
        total=len(snp_chunks),
    ):
        if method == "HET":
            test_func = _block_het_test
        else:
            test_func = _block_test
        # test each SNP in block
        res.append(
            test_func(
                var=var[:, snp_start * var_size : snp_stop * var_size],
                cov=cov,
                pheno=pheno,
                var_size=var_size,
                test_vars=test_vars,
                family=family,
                fast=fast,
            )
        )
    # columns will be BETA1, SE1, BETA2, SE2, ... with n_anc
    columns = list(
        itertools.chain.from_iterable([[f"{v}_BETA", f"{v}_SE"] for v in var_names])
    ) + ["N", "P"]

    df_res = pd.DataFrame(
        np.concatenate(res),
        columns=columns,
        index=dset.snp.index if dset is not None else None,
    ).astype({"N": "int"})
    df_res.loc[df_res.P.isna(), df_res.columns != "N"] = np.nan

    return df_res


def marginal_simple(dset: admix.Dataset, pheno: np.ndarray) -> np.ndarray:
    """Simple marginal association testing for one SNP at a time

    Useful in simulation study because this will be very fast

    Parameters
    ----------
    dset : admix.Dataset
        Dataset containing the (n_indiv, n_snp) genotype matrix, dset.geno
    pheno : np.ndarray
        (n_snp, n_sim) phenotype matrix

    Returns
    -------
    coef : np.ndarray
        (n_snp, n_sim) marginal association coefficient
    coef_se: np.ndarray
        (n_snp, n_sim) marginal association coefficient standard error
    zscores : np.ndarray
        (n_snp, n_sim) association z-scores for each SNP being tested

    Examples
    --------

    To check the consistency of results of standard methods

    >>> n_indiv = dset_admix.dims["indiv"]
    >>> n_cov = 1

    >>> geno = _impute_with_mean(dset_admix.geno.values)
    >>> geno = (geno - geno.mean(axis=0)) / geno.std(axis=0)

    >>> f_stats = tinygwas.linear_f_test(geno, np.ones((n_indiv, 1)), sim["pheno"][:, 0], 1, [0])
    >>> p_vals = stats.f.sf(f_stats, 1, n_indiv - n_cov - 1)
    >>> zscores2 = stats.norm.ppf(p_vals / 2) * np.sign(zscores[:, 0])

    >>> dset = admix.Dataset({"geno": (["indiv", "snp"], geno), "pheno": (["snp", "sim"], pheno)})
    >>> zscores = marginal_simple(dset, pheno)

    """
    geno = dset.geno
    n_indiv, n_snp = geno.shape
    assert (
        n_indiv == pheno.shape[0]
    ), "Number of individuals in genotype and phenotype do not match"
    n_sim = pheno.shape[1]

    # center phenotype for each simulation
    Y = pheno - pheno.mean(axis=0)
    X = geno - da.nanmean(geno, axis=0)

    XtY, snp_var = admix.data.geno_mult_mat(
        X, Y, transpose_geno=True, return_snp_var=True
    )
    XtX = snp_var * n_indiv

    coef = XtY / XtX[:, np.newaxis]
    coef_var = np.var(Y, axis=0) / XtX[:, np.newaxis]
    coef_se = np.sqrt(coef_var)

    zscores = coef / coef_se

    return coef, coef_se, zscores


# def mixscore_wrapper(pheno, anc, geno, theta,
#                     scores=["ADM", "ATT", "MIX", "SNP1", "SUM"],
#                     mixscore_path="/u/project/pasaniuc/kangchen/tractor/software/mixscore-1.3/bin/mixscore",
#                     verbose=False):
#     """
#     A python wrapper for mixscore
#
#     Args
#     ----
#     pheno: phenotypes
#     anc: ancestry
#     geno: genotype
#     theta: global ancestry component
#     """
#
#     tmp = tempfile.TemporaryDirectory()
#     tmp_dir = tmp.name
#
#     n_sample = len(pheno)
#     n_snp = anc.shape[1]
#
#     write_int_mat(join(tmp_dir, "pheno"), pheno.reshape((1, -1)))
#     write_int_mat(join(tmp_dir, "anc"), anc.T)
#     write_int_mat(join(tmp_dir, "geno"), geno.T)
#     np.savetxt(join(tmp_dir, "theta"), theta, fmt='%.6f')
#
#     param = {"nsamples": str(n_sample),
#               "nsnps": str(n_snp),
#               "phenofile": join(tmp_dir, "pheno"),
#               "ancfile": join(tmp_dir, "anc"),
#               "genofile": join(tmp_dir, "geno"),
#               "thetafile": join(tmp_dir, "theta"),
#               "outfile": join(tmp_dir, "out")}
#
#     with open(join(tmp_dir, "param"), 'w') as f:
#         f.writelines([k + ':' + param[k] + '\n' for k in param])
#
#     rls_dict = {}
#     for name in scores:
#         if verbose:
#             print(f"Calculating {name}...")
#
#         cmd = ' '.join([mixscore_path, name, f"{tmp_dir}/param"])
#         subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
#         with open(param["outfile"]) as f:
#             out = [line.strip() for line in f.readlines()]
#         rls_dict[name] = out
#     tmp.cleanup()
#     score_df = pd.DataFrame(rls_dict).apply(pd.to_numeric, errors='coerce')
#     # convert to p-value
#     for name in score_df.columns:
#         score_df[name] = stats.chi2.sf(score_df[name], df=(2 if name == "SUM" else 1))
#     return score_df

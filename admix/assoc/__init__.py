import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import admix
from typing import Any, Dict, List, Tuple, Optional, Union
import dask.array as da
import admix
from statsmodels.tools import sm_exceptions
import warnings

warnings.filterwarnings(action="error", category=sm_exceptions.ValueWarning)

__all__ = ["marginal", "marginal_simple"]

# TODO: merge marginal_fast and marginal


def _block_test(
    var: np.ndarray,
    cov: np.ndarray,
    pheno: np.ndarray,
    var_size: int,
    test_vars: List[int],
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

    Todo
    ----
    TODO: what happens when the covariates perfectly correlate?
    TODO: also return effect sizes in additional to p-values
    """
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
    ), "Number of variables in var must be a multiple of var_size"
    n_var = var.shape[1] // var_size

    test_vars = np.array(test_vars)
    assert np.all(test_vars < var_size), "test_vars must be less than var_size"

    n_cov = cov.shape[1]
    design = np.zeros((n_indiv, var_size + n_cov))

    design[:, var_size : var_size + n_cov] = cov
    if fast:
        try:
            import tinygwas
        except ImportError:
            raise ImportError(
                "\nplease install tinygwas:\n\n"
                "\tpip install git+https://github.com/bogdanlab/tinygwas.git#egg=tinygwas"
            )

        if family == "linear":
            # number of non-nan individuals per test
            f_stats = np.empty(n_var)
            n_indiv_test = np.empty(n_var)
            # tinygwas return fstats and nindiv through reference
            tinygwas.linear_f_test(
                var, cov, pheno, var_size, test_vars, f_stats, n_indiv_test
            )
            pvalues = stats.f.sf(
                f_stats, len(test_vars), n_indiv_test - n_cov - var_size
            )

        elif family == "logistic":
            if "max_iter" not in logistic_kwargs:
                logistic_kwargs["max_iter"] = 100
            if "tol" not in logistic_kwargs:
                logistic_kwargs["tol"] = 1e-6

            lrt_diff = tinygwas.logistic_lrt(
                var,
                cov,
                pheno,
                var_size,
                test_vars,
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            pvalues = stats.chi2.sf(2 * lrt_diff, len(test_vars))
        else:
            raise ValueError(f"Unknown family: {family}")
    else:
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

        pvalues = []

        reduced_index = np.concatenate(
            [
                [i for i in range(var_size) if i not in test_vars],
                np.arange(var_size, var_size + n_cov),
            ]
        ).astype(int)

        f_test_r_matrix = np.zeros((len(test_vars), design.shape[1]))
        for i, v in enumerate(test_vars):
            f_test_r_matrix[i, v] = 1
        for i in range(n_var):
            design[:, 0:var_size] = var[:, i * var_size : (i + 1) * var_size]
            model = reg_method(pheno, design)
            if family == "linear":
                # f-test using statsmodels
                try:
                    pvalues.append(model.f_test(f_test_r_matrix).pvalue.item())
                except sm_exceptions.ValueWarning:
                    pvalues.append(np.nan)
            elif family == "logistic":
                # more than one test variables
                model_reduced = reg_method(
                    pheno,
                    design[:, reduced_index],
                    start_params=model.params[reduced_index],
                )
                # determine p-values using difference in log-likelihood
                # and difference in degrees of freedom
                pvalues.append(
                    stats.chi2.sf(
                        -2 * (model_reduced.llf - model.llf),
                        (model.df_model - model_reduced.df_model),
                    )
                )
    return np.array(pvalues)


def marginal(
    pheno: np.ndarray,
    dset: admix.Dataset = None,
    geno: da.Array = None,
    lanc: da.Array = None,
    n_anc: Optional[int] = None,
    cov: Optional[np.ndarray] = None,
    method: str = "ATT",
    family: str = "linear",
    verbose: bool = False,
    fast: bool = True,
):
    """Marginal association testing for one SNP at a time

    Parameters
    ----------
    dset : admix.Dataset
        [description]
    pheno : str
        [description]
    cov : List[str], optional
        [description], by default None
    method : str, optional
        [description], by default "ATT"
    family : str, optional
        [description], by default "linear"

    Returns
    -------
    np.ndarray
        Association p-values for each SNP being tested

    """
    if family == "logistic":
        admix.logger.warn("logistic family is not well tested, use with caution")
    # format data
    assert method in ["ATT", "TRACTOR", "ADM", "SNP1", "ASE"]
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
        assert np.all(
            np.nanmean(cov, axis=1) == debug_old_mean
        ), "NaN imputation failed"

    # check covariates must be full rank
    assert np.linalg.matrix_rank(cov) == cov.shape[1], "Covariates must be of full rank"

    if method == "ATT":
        var = geno.sum(axis=2).swapaxes(0, 1)
        var_size = 1
        test_vars = [0]
    elif method == "TRACTOR":
        allele_per_anc = admix.data.allele_per_anc(
            geno,
            lanc,
            n_anc=n_anc,
        ).swapaxes(0, 1)
        lanc = lanc.sum(axis=2).swapaxes(0, 1)
        var = da.empty((n_indiv, n_snp * 3))
        var[:, 0::3] = allele_per_anc[:, :, 0]
        var[:, 1::3] = allele_per_anc[:, :, 1]
        var[:, 2::3] = lanc
        var_size = 3
        test_vars = [0, 1]
    elif method == "SNP1":
        geno = geno.sum(axis=2).swapaxes(0, 1)
        lanc = lanc.sum(axis=2).swapaxes(0, 1)
        var = da.empty((n_indiv, n_snp * 2))
        var[:, 0::2] = geno
        var[:, 1::2] = lanc
        var_size = 2
        test_vars = [0]
    elif method == "ASE":
        # alleles per ancestry
        allele_per_anc = admix.data.allele_per_anc(geno, lanc, n_anc=n_anc).swapaxes(
            0, 1
        )
        var = da.empty((n_indiv, n_snp * 2))
        var[:, 0::2] = allele_per_anc[:, :, 0]
        var[:, 1::2] = allele_per_anc[:, :, 1]
        var_size = 2
        test_vars = [0, 1]
    elif method == "ADM":
        var = lanc.sum(axis=2).swapaxes(0, 1)
        var_size = 1
        test_vars = [0]
    else:
        raise NotImplementedError

    # iterate over block of SNPs
    assert var.shape[1] % var_size == 0, "var must have multiple of `var_size` columns"
    assert var.shape[1] / var_size == n_snp

    pvalues = []
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
        # test each SNP in block
        pvalues.append(
            _block_test(
                var=var[:, snp_start * var_size : snp_stop * var_size],
                cov=cov,
                pheno=pheno,
                var_size=var_size,
                test_vars=test_vars,
                family=family,
                fast=fast,
            )
        )
    return np.concatenate(pvalues)


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

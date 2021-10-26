import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from typing import List
from tqdm import tqdm
import admix
from typing import Any, Dict
import dask.array as da
import admix

__all__ = ["marginal", "marginal_fast", "marginal_simple"]


def marginal_fast(
    dset: admix.Dataset,
    pheno_col: str,
    cov_cols: List[str] = None,
    method: str = "ATT",
    family: str = "linear",
    logistic_kwargs: Dict[str, Any] = dict(),
    verbose: bool = False,
):
    """Marginal association testing for one SNP at a time

    Parameters
    ----------
    dset : xr.Dataset
        [description]
    pheno : str
        [description]
    cov : List[str], optional
        [description], by default None
    family : str, optional
        distribution assumption of response variable, one of "linear" and "logistic",
        by default "linear"
    method : str, optional
        method to perform GWAS, one of "ATT" / "TRACTOR" / "ADM" / "SNP1"
        by default "ATT"

    Returns
    -------
    [type]
        Association p-values for each SNP being tested

    Raises
    ------
    NotImplementedError
        [description]
    NotImplementedError
        [description]

    Todo
    ----
    TODO: what happens when the covariates perfectly correlate?
    TODO: iterate SNPs by chunk and contatenate results for all chunks
    TODO: also return effect sizes in additional to p-values
    """

    try:
        import admixgwas
    except ImportError:
        raise ImportError("\nplease install admixgwas:\n\n\tpip install admixgwas")

    assert family in ["linear", "logistic"]
    assert method in ["ATT", "TRACTOR", "ADM", "SNP1"]

    if family == "logistic":
        if "max_iter" not in logistic_kwargs:
            logistic_kwargs["max_iter"] = 200
        if "tol" not in logistic_kwargs:
            logistic_kwargs["tol"] = 1e-6

    n_indiv = dset.n_indiv
    n_snp = dset.n_snp
    pheno = dset.indiv[pheno_col].values
    mask_indiv = ~np.isnan(pheno)
    if cov_cols is not None:
        cov_values = np.column_stack(
            [np.ones(n_indiv)] + [dset.indiv[col].data for col in cov_cols]
        )
    else:
        cov_values = np.ones((n_indiv, 1))

    n_cov = cov_values.shape[1]
    if method == "ATT":
        # [geno]
        geno = np.swapaxes(np.sum(dset.geno, axis=2), 0, 1).compute()

        if family == "linear":
            f_stats = admixgwas.linear_f_test(geno, cov_values, pheno, 1, [0])
            p_vals = stats.f.sf(f_stats, 1, n_indiv - n_cov - 1)
        elif family == "logistic":
            lrt_diff = admixgwas.logistic_lrt(
                geno,
                cov_values,
                pheno,
                1,
                [0],
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            p_vals = stats.chi2.sf(2 * lrt_diff, 1)
        else:
            raise NotImplementedError

    elif method == "SNP1":
        # [geno] + lanc
        geno = np.swapaxes(np.sum(dset.geno, axis=2), 0, 1).compute()
        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1).compute()
        var = np.empty((geno.shape[0], n_snp * 2))
        var[:, 0::2] = geno
        var[:, 1::2] = lanc
        if family == "linear":
            f_stats = admixgwas.linear_f_test(var, cov_values, pheno, 2, [0])
            p_vals = stats.f.sf(f_stats, 1, n_indiv - n_cov - 2)
        elif family == "logistic":
            lrt_diff = admixgwas.logistic_lrt(
                var,
                cov_values,
                pheno,
                2,
                [0],
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            p_vals = stats.chi2.sf(2 * lrt_diff, 1)
        else:
            raise NotImplementedError
    elif method == "TRACTOR":
        # lanc + [allele1 + allele2]
        # number of african alleles
        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1).compute()
        dset.compute_allele_per_anc()
        # alleles per ancestry
        allele_per_anc = np.swapaxes(dset.allele_per_anc.compute(), 0, 1)

        var = np.empty((n_indiv, n_snp * 3))
        var[:, 0::3] = allele_per_anc[:, :, 0]
        var[:, 1::3] = allele_per_anc[:, :, 1]
        var[:, 2::3] = lanc
        if family == "linear":
            f_stats = admixgwas.linear_f_test(var, cov_values, pheno, 3, [0, 1])
            p_vals = stats.f.sf(f_stats, 2, n_indiv - n_cov - 3)
        elif family == "logistic":
            lrt_diff = admixgwas.logistic_lrt(
                var,
                cov_values,
                pheno,
                3,
                [0, 1],
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            p_vals = stats.chi2.sf(2 * lrt_diff, 2)
        else:
            raise NotImplementedError

    elif method == "ADM":
        # [lanc]
        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1)
        if family == "linear":
            f_stats = admixgwas.linear_f_test(lanc, cov_values, pheno, 1, [0])
            p_vals = stats.f.sf(f_stats, 1, n_indiv - n_cov - 1)
        elif family == "logistic":
            lrt_diff = admixgwas.logistic_lrt(
                lanc,
                cov_values,
                pheno,
                1,
                [0],
                logistic_kwargs["max_iter"],
                logistic_kwargs["tol"],
            )
            p_vals = stats.chi2.sf(2 * lrt_diff, 1)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return pd.DataFrame({"SNP": dset.snp.index.values, "P": p_vals}).set_index("SNP")


def marginal(
    dset: admix.Dataset,
    pheno_col: str,
    cov_cols: List[str] = None,
    method: str = "ATT",
    family: str = "linear",
    verbose: bool = False,
):
    """Marginal association testing for one SNP at a time

    Parameters
    ----------
    dset : xr.Dataset
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
    [type]
        Association p-values for each SNP being tested

    Raises
    ------
    NotImplementedError
        [description]
    NotImplementedError
        [description]
    """

    if family == "linear":
        glm_family = sm.families.Gaussian()
    elif family == "logistic":
        glm_family = sm.families.Binomial()
    else:
        raise NotImplementedError

    assert method in ["ATT", "TRACTOR", "ADM", "SNP1"]

    pheno = dset.indiv[pheno_col]
    n_snp = dset.n_snp
    mask_indiv = ~np.isnan(pheno)
    if cov_cols is not None:
        cov = np.vstack([dset.indiv[col] for col in cov_cols]).T
    else:
        cov = np.array([], dtype=np.int64).reshape(dset.n_indiv, 0)

    # TODO: deal with missing genotype (current it is fine because of imputation)
    if method == "ATT":
        geno = np.swapaxes(np.sum(dset.geno, axis=2), 0, 1)
        pvalues = []
        for i_snp in tqdm(range(n_snp), disable=not verbose):
            design = np.hstack([sm.add_constant(geno[:, i_snp][:, np.newaxis]), cov])
            if family == "linear":
                model = sm.OLS(pheno[mask_indiv], design[mask_indiv, :]).fit(disp=0)
            elif family == "logistic":
                model = sm.Logit(pheno[mask_indiv], design[mask_indiv, :]).fit(disp=0)
            pvalues.append(model.pvalues[1])

        pvalues = np.array(pvalues)

    elif method == "SNP1":
        geno = np.swapaxes(np.sum(dset.geno, axis=2), 0, 1).compute()
        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1).compute()

        pvalues = []
        for i_snp in tqdm(range(n_snp), disable=not verbose):

            design = np.hstack(
                [
                    sm.add_constant(geno[:, i_snp][:, np.newaxis]),
                    lanc[:, i_snp][:, np.newaxis],
                    cov,
                ]
            )
            model = sm.GLM(
                pheno[mask_indiv], design[mask_indiv, :], family=glm_family
            ).fit()
            pvalues.append(model.pvalues[1])
        pvalues = np.array(pvalues)

    elif method == "TRACTOR":

        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1).compute()
        dset.compute_allele_per_anc()
        # alleles per ancestry
        allele_per_anc = np.swapaxes(dset.allele_per_anc.compute(), 0, 1)

        pvalues = []
        for i_snp in tqdm(range(n_snp), disable=not verbose):
            # number of african alleles, covariates
            design_null = np.hstack(
                [sm.add_constant(lanc[:, i_snp][:, np.newaxis]), cov]
            )
            model_null = sm.GLM(
                pheno[mask_indiv],
                design_null[mask_indiv, :],
                family=glm_family,
            ).fit()
            # number of african alleles, covariates + allele-per-anc
            design_alt = np.hstack([design_null, allele_per_anc[:, i_snp, :]])
            model_alt = sm.GLM(
                pheno[mask_indiv],
                design_alt[mask_indiv, :],
                family=glm_family,
            ).fit(start_params=np.concatenate([model_null.params, [0.0, 0.0]]))
            pvalues.append(stats.chi2.sf(-2 * (model_null.llf - model_alt.llf), 2))
        pvalues = np.array(pvalues)

    elif method == "ADM":
        lanc = np.swapaxes(np.sum(dset.lanc, axis=2), 0, 1).compute()
        pvalues = []
        for i_snp in tqdm(range(n_snp), disable=not verbose):
            design = np.hstack([sm.add_constant(lanc[:, i_snp][:, np.newaxis]), cov])
            model = sm.GLM(
                pheno[mask_indiv], design[mask_indiv, :], family=sm.families.Gaussian()
            ).fit()
            pvalues.append(model.pvalues[1])
        pvalues = np.array(pvalues)

    else:
        raise NotImplementedError

    return pd.DataFrame({"SNP": dset.snp.index.values, "P": pvalues}).set_index("SNP")


def marginal_simple(dset: xr.Dataset, pheno: np.ndarray) -> np.ndarray:
    """Simple marginal association testing for one SNP at a time

    Useful in simulation study because this will be very fast

    Parameters
    ----------
    dset : xr.Dataset
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

    >>> f_stats = admixgwas.linear_f_test(geno, np.ones((n_indiv, 1)), sim["pheno"][:, 0], 1, [0])
    >>> p_vals = stats.f.sf(f_stats, 1, n_indiv - n_cov - 1)
    >>> zscores2 = stats.norm.ppf(p_vals / 2) * np.sign(zscores[:, 0])

    >>> dset = xr.Dataset({"geno": (["indiv", "snp"], geno), "pheno": (["snp", "sim"], pheno)})
    >>> zscores = marginal_simple(dset, pheno)

    """
    geno = dset["geno"].data
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

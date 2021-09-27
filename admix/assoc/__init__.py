import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from typing import List
from tqdm import tqdm
import admix
from typing import Any, Dict

__all__ = ["marginal", "marginal_fast"]


def marginal_fast(
    dset: xr.Dataset,
    pheno: str,
    cov: List[str] = None,
    method: str = "ATT",
    family: str = "linear",
    logistic_kwargs: Dict[str, Any] = dict(),
    verbose: bool = False,
):
    """Marginal association testing for one SNP at a time
    TODO: what happens when the covariates perfectly correlate?
    TODO: iterate SNPs by chunk and contatenate results for all chunks
    TODO: also return effect sizes in additional to p-values

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

    n_indiv = dset.dims["indiv"]

    pheno = dset[pheno].data
    n_snp = dset.dims["snp"]
    mask_indiv = ~np.isnan(pheno)
    if cov is not None:
        cov_values = np.column_stack(
            [np.ones(n_indiv)] + [dset[col].data for col in cov]
        )
    else:
        cov_values = np.ones((n_indiv, 1))

    n_cov = cov_values.shape[1]
    if method == "ATT":
        # [geno]
        geno = np.sum(dset["geno"].data, axis=2)

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
            p_vals = stats.chi2.sf(-2 * lrt_diff, 1)
        else:
            raise NotImplementedError

    elif method == "SNP1":
        # [geno] + lanc
        geno = np.sum(dset["geno"].data, axis=2)
        lanc = np.sum(dset["lanc"].data, axis=2)
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
            p_vals = stats.chi2.sf(-2 * lrt_diff, 1)
        else:
            raise NotImplementedError
    elif method == "TRACTOR":
        # lanc + [allele1 + allele2]
        # number of african alleles
        lanc = np.sum(dset["lanc"].data, axis=2)
        # alleles per ancestry
        allele_per_anc = admix.tools.allele_per_anc(dset, inplace=False).compute()

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
            p_vals = stats.chi2.sf(-2 * lrt_diff, 2)
        else:
            raise NotImplementedError

    elif method == "ADM":
        # [lanc]
        lanc = np.sum(dset["lanc"].data, axis=2)
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
            p_vals = stats.chi2.sf(-2 * lrt_diff, 1)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return pd.DataFrame({"SNP": dset.snp.values, "P": p_vals}).set_index("SNP")


def marginal(
    dset: xr.Dataset,
    pheno: str,
    cov: List[str] = None,
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

    pheno = dset[pheno].data
    n_snp = dset.dims["snp"]
    mask_indiv = ~np.isnan(pheno)
    if cov is not None:
        cov = np.vstack([dset[col].data for col in cov]).T
    else:
        cov = np.array([], dtype=np.int64).reshape(dset.dims["indiv"], 0)

    # TODO: deal with missing genotype (current it is fine because of imputation)
    if method == "ATT":
        geno = np.sum(dset["geno"].data, axis=2)
        pvalues = []
        for i_snp in tqdm(range(n_snp), disable=not verbose):
            design = np.hstack([sm.add_constant(geno[:, i_snp][:, np.newaxis]), cov])
            if family == "linear":
                model = sm.OLS(pheno[mask_indiv], design[mask_indiv, :]).fit()
            elif family == "logistic":
                model = sm.Logit(pheno[mask_indiv], design[mask_indiv, :]).fit()
            pvalues.append(model.pvalues[1])
            # model = sm.GLM(
            #     pheno[mask_indiv], design[mask_indiv, :], family=glm_family
            # ).fit()
        pvalues = np.array(pvalues)

    elif method == "SNP1":
        geno = np.sum(dset["geno"].data, axis=2)
        lanc = np.sum(dset["lanc"].data, axis=2)

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
        # number of african alleles
        lanc = np.sum(dset["lanc"].data, axis=2)
        # alleles per ancestry
        allele_per_anc = admix.tools.allele_per_anc(dset, inplace=False).compute()

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
        lanc = np.sum(dset["lanc"].data, axis=2)
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

    return pd.DataFrame({"SNP": dset.snp.values, "P": pvalues}).set_index("SNP")


def logistic_reg(X, y, cov):

    pass


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

import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
from typing import List
from tqdm import tqdm
from admix.data import compute_allele_per_anc


__all__ = ["marginal", "linear_reg"]


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
            model = sm.GLM(
                pheno[mask_indiv], design[mask_indiv, :], family=glm_family
            ).fit()
            pvalues.append(model.pvalues[1])
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
        allele_per_anc = compute_allele_per_anc(dset).compute()

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


def linear_reg(X: np.ndarray, y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Implement linear regression for numpy matrix

    Refer to `f_regression_cov` in FaST-LMM package
    TODO: check `f_regression_cov_alt` in FaST-LMM package to see the difference
    TODO: double check math to see the difference.
    TODO: what happens when the covariates perfectly correlate?

    Args:
        X (np.ndarray): (n_indiv, n_snp) genotype matrix
        y (np.ndarray): (n_indiv, ) phenotype matrix
        cov (np.ndarray): (n_indiv, n_cov) covariance matrix

    Returns:
        np.ndarray: p-values for each SNP
    """

    n_snp = X.shape[1]

    assert cov.shape[0] > cov.shape[1]

    cov_pinv = np.linalg.pinv(cov)
    X -= np.dot(cov, (np.dot(cov_pinv, X)))
    y -= np.dot(cov, (np.dot(cov_pinv, y)))

    # compute the correlation
    corr = np.dot(y, X)
    corr /= np.asarray(np.sqrt((X ** 2).sum(axis=0))).ravel()
    corr /= np.asarray(np.sqrt((y ** 2).sum())).ravel()

    # convert to p-value
    dof = (X.shape[0] - 1 - cov.shape[1]) / (1)  # (df_fm / (df_rm - df_fm))
    F = corr ** 2 / (1 - corr ** 2) * dof
    pv = stats.f.sf(F, 1, dof)
    return pv


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

import statsmodels.api as sm
import numpy as np
import pandas as pd
from os.path import join
from scipy import stats
import xarray as xr
from typing import List
from tqdm import tqdm
from admix.data import compute_allele_per_anc


__all__ = ["marginal"]


def marginal(
    dset: xr.Dataset,
    pheno: str,
    cov: List[str] = None,
    method: str = "ATT",
    family: str = "gaussian",
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
        [description], by default "gaussian"

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

    if family == "gaussian":
        glm_family = sm.families.Gaussian()
    elif family == "binomial":
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
        for i_snp in tqdm(range(n_snp)):
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
        for i_snp in tqdm(range(n_snp)):

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
        for i_snp in tqdm(range(n_snp)):
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
        for i_snp in tqdm(range(n_snp)):
            design = np.hstack([sm.add_constant(lanc[:, i_snp][:, np.newaxis]), cov])
            model = sm.GLM(
                pheno[mask_indiv], design[mask_indiv, :], family=sm.families.Gaussian()
            ).fit()
            pvalues.append(model.pvalues[1])
        pvalues = np.array(pvalues)

    else:
        raise NotImplementedError

    return pd.DataFrame({"SNP": dset.snp.values, "P": pvalues}).set_index("SNP")


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

import admix
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import glob
from natsort import natsorted
import os
import json
from scipy import stats
from ._utils import log_params
from scipy.interpolate import CubicSpline


def _write_admix_grm(
    dict_grm: Dict[str, np.ndarray],
    df_weight: pd.Series,
    df_id: pd.DataFrame,
    out_prefix: str,
):
    """write admix grm

    Parameters
    ----------
    dict_grm : Dict[str, np.ndarray]
        name -> grm matrix
    df_weight : pd.Series
        snp weights
    n_snp : int
        number of snps
    df_id: pd.DataFrame
        list of individuals
    out_prefix: str
        prefix of the output file

    Returns
    -------
    None
    """
    for name in dict_grm:
        grm = dict_grm[name]
        admix.tools.gcta.write_grm(
            out_prefix + "." + name,
            K=grm,
            df_id=df_id,
            n_snps=np.repeat(len(df_weight), len(df_id)),
        )

    # write weight
    df_weight.to_csv(out_prefix + ".weight.tsv", sep="\t")


# Implementing genetic correlation related functions
def admix_grm(
    pfile: str,
    out_prefix: str,
    maf_cutoff: float = 0.005,
    her_model="mafukb",
    freq_cols=["LANC_FREQ1", "LANC_FREQ2"],
    snp_chunk_size: int = 256,
    snp_list: str = None,
    write_raw: bool = False,
) -> None:
    """
    Calculate the admix GRM for a given pfile

    Parameters
    ----------
    pfile : str
        Path to the pfile
    out_prefix : str
        Prefix of the output files
    maf_cutoff : float, optional
        MAF cutoff for the admixed individuals, by default 0.005
    her_model : str, optional
        Heritability model, by default "mafukb"
        one of "uniform", "gcta", "ldak", "mafukb"
    freq_cols : List[str], optional
        Columns of the pfile to use as frequency, by default ["LANC_FREQ1", "LANC_FREQ2"]
        to perform the ancestry-specific MAF cutoffs
    snp_chunk_size : int, optional
        Number of SNPs to read at a time, by default 256
        This can be tuned to reduce memory usage
    snp_list : str, optional
        Path to a file containing a list of SNPs to use. Each line should be a SNP ID.
        Only SNPs in the list will be used for the analysis. By default None
    write_raw: bool, optional
        Whether to write the raw GRM, G1, G2, G12, by default False
    Returns
    -------
    GRM files: {out_prefix}.[K1, K2].[grm.bin | grm.id | grm.n] will be generated
    Weight file: {out_prefix}.weight.tsv will be generated
    """

    log_params("admix-grm", locals())
    assert len(freq_cols) == 2, "freq_cols must be a list of length 2"
    dset = admix.io.read_dataset(pfile=pfile, snp_chunk=snp_chunk_size)

    # filter for SNPs
    if snp_list is not None:
        with open(snp_list, "r") as f:
            filter_snp_list = [line.strip() for line in f]
        n_filter_snp = len(filter_snp_list)
        filter_snp_list = dset.snp.index[dset.snp.index.isin(filter_snp_list)]
        if len(filter_snp_list) < n_filter_snp:
            admix.logger.warning(
                f"{n_filter_snp - len(filter_snp_list)} SNPs in {snp_list} are not in the dataset"
            )
        dset = dset[filter_snp_list]

    snp_subset = np.where(
        dset.snp[freq_cols[0]].between(maf_cutoff, 1 - maf_cutoff)
        & dset.snp[freq_cols[1]].between(maf_cutoff, 1 - maf_cutoff)
    )[0]

    dset = dset[snp_subset]

    dset.snp["PRIOR_VAR"] = admix.data.calc_snp_prior_var(dset.snp, her_model=her_model)

    K1, K2 = admix.data.admix_grm_equal_var(
        geno=dset.geno,
        lanc=dset.lanc,
        snp_prior_var=dset.snp.PRIOR_VAR.values,
        n_anc=dset.n_anc,
    )

    df_weight = dset.snp.PRIOR_VAR
    df_id = pd.DataFrame({"0": dset.indiv.index.values, "1": dset.indiv.index.values})
    _write_admix_grm(
        dict_grm={"K1": K1, "K2": K2},
        df_weight=df_weight,
        df_id=df_id,
        out_prefix=out_prefix,
    )

    if write_raw:
        G1, G2, G12 = admix.data.admix_grm(
            geno=dset.geno,
            lanc=dset.lanc,
            snp_prior_var=dset.snp.PRIOR_VAR.values,
        )

        for suffix, mat in zip(["G1", "G2", "G12"], [G1, G2, G12]):
            admix.tools.gcta.write_grm(
                f"{out_prefix}.{suffix}",
                K=mat,
                df_id=df_id,
                n_snps=np.repeat(len(df_weight), len(df_id)),
            )


def admix_grm_merge(prefix: str, out_prefix: str, n_part: int = 22) -> None:
    """
    Merge multiple GRM matrices

    Parameters
    ----------
    prefix : str
        Prefix of the GRM files, any files with the pattern of <prefix>.*
        will be merged
    out_prefix : str
        Prefix of the output file
    n_part : int, optional
        Number of partitions, by default 22

    Returns
    -------
    GRM files: {out_prefix}.[K1, K2].[grm.bin | grm.id | grm.n] will be generated
    Weight file: {out_prefix}.weight.tsv will be generated
    """
    log_params("admix-grm-merge", locals())

    # search for files with the pattern of <prefix>.<suffix>.K1.grm.bin and <prefix>.<suffix>.K2.grm.bin
    K1_grm_prefix_list = [
        f[: -len(".K1.grm.bin")] for f in sorted(glob.glob(f"{prefix}*.K1.grm.bin"))
    ]
    K2_grm_prefix_list = [
        f[: -len(".K2.grm.bin")] for f in sorted(glob.glob(f"{prefix}*.K2.grm.bin"))
    ]
    assert len(K1_grm_prefix_list) == len(K2_grm_prefix_list)
    assert K1_grm_prefix_list == K2_grm_prefix_list, (
        "GRM files .K1 and .K2 are not matched, "
        f"K1_grm_list={K1_grm_prefix_list}, K2_grm_list={K2_grm_prefix_list}"
    )
    grm_prefix_list = natsorted(K1_grm_prefix_list)
    if len(grm_prefix_list) != n_part:
        raise ValueError(
            f"Number of GRM files ({len(grm_prefix_list)}) is not equal to n_part ({n_part})"
        )
    admix.logger.info(f"{len(grm_prefix_list)} GRM files to be merged: {grm_prefix_list}")
    prior_var_list = []
    for grm_prefix in grm_prefix_list:
        prior_var_list.append(pd.read_csv(grm_prefix + ".weight.tsv", sep="\t", index_col=0))

    def _merge(suffix):
        total_grm = 0
        total_df_id = None

        weight_list = []
        for i, grm_prefix in enumerate(grm_prefix_list):
            prior_var = prior_var_list[i]
            weight = prior_var["PRIOR_VAR"].sum()
            weight_list.append(weight)
            grm, df_id, n_snps = admix.tools.gcta.read_grm(f"{grm_prefix}.{suffix}")
            if total_df_id is None:
                total_df_id = df_id
            else:
                assert np.all(
                    total_df_id == df_id
                ), f"df_id of {grm_prefix} is not matched does not match with already loaded ones"
            total_grm += grm * weight

            assert np.all(n_snps == prior_var.shape[0])
        total_grm /= np.sum(weight_list)
        return total_grm, df_id

    K1, df_id1 = _merge("K1")
    K2, df_id2 = _merge("K2")
    assert df_id1.equals(df_id2)
    df_id = df_id1
    df_weight = pd.concat(prior_var_list)

    _write_admix_grm(
        dict_grm={"K1": K1, "K2": K2},
        df_weight=df_weight,
        df_id=df_id,
        out_prefix=out_prefix,
    )


def genet_cor(
    pheno: str,
    grm_prefix: str,
    out_dir: str,
    rg_grid=np.linspace(0, 1.0, 21),
    quantile_normalize: bool = True,
    n_thread: int = 2,
    clean: bool = True,
):
    """Estimate genetic correlation

    Parameters
    ----------
    pheno : str
        phenotype file, the 1st column contains ID, 2nd column contains phenotype, and
        the rest of columns are covariates.
    grm_prefix : str
        folder containing K1, K2 GRM files
    out_dir : str
        folder to store the output files
    rg_grid : list, optional
        List of rg values to grid search, by default np.linspace(0, 1.0, 21)
    quantile_normalize: bool
        whether to perform quantile normalization for both phenotype and each column of covariates
    n_thread : int, optional
        number of threads, by default 2
    """
    log_params("genet-cor", locals())

    ## compile phenotype and covariates
    df_pheno = pd.read_csv(pheno, sep="\t", index_col=0)
    df_pheno.index = df_pheno.index.astype(str)

    # subset for individuals with non-nan value in df_trait
    trait_col = df_pheno.columns[0]
    covar_cols = df_pheno.columns[1:]

    # filter out individuals with missing phenotype
    df_pheno = df_pheno[df_pheno[trait_col].notna()]

    df_trait = df_pheno[[trait_col]].copy()
    df_covar = df_pheno[covar_cols].copy()
    df_covar = admix.data.convert_dummy(df_covar)
    if quantile_normalize:
        # perform quantile normalization
        for col in df_trait.columns:
            df_trait[col] = admix.data.quantile_normalize(df_trait[col])

        for col in df_covar.columns:
            df_covar[col] = admix.data.quantile_normalize(df_covar[col])

    # fill na with column mean
    df_covar.fillna(df_covar.mean(), inplace=True)

    df_id = pd.DataFrame(
        {"FID": df_trait.index.values, "IID": df_trait.index.values},
        index=df_trait.index.values,
    )
    df_trait = pd.merge(df_id, df_trait, left_index=True, right_index=True)
    df_covar = pd.merge(df_id, df_covar, left_index=True, right_index=True)

    ## load grm
    K1, df_id1, n_snps1 = admix.tools.gcta.read_grm(grm_prefix + ".K1")
    K2, df_id2, n_snps2 = admix.tools.gcta.read_grm(grm_prefix + ".K2")
    assert df_id1.equals(df_id2)
    assert np.allclose(n_snps1, n_snps2)
    df_id = df_id1
    n_snps = n_snps1

    os.makedirs(out_dir, exist_ok=True)
    for rg in tqdm(rg_grid):
        K = K1 + K2 * rg

        grm = os.path.join(out_dir, f"rg{int(rg * 100)}")
        admix.tools.gcta.write_grm(
            grm,
            K=K,
            df_id=df_id,
            n_snps=n_snps,
        )
        admix.tools.gcta.reml(
            grm_path=grm,
            df_pheno=df_trait,
            df_covar=df_covar,
            out_prefix=os.path.join(out_dir, f"rg{int(rg * 100)}"),
            n_thread=n_thread,
            est_fix=True,
        )
        if clean:
            # remove <grm>.grm.* files
            for f in glob.glob(grm + ".grm.*"):
                os.remove(f)


def summarize_genet_cor(
    est_dir: str,
    out_prefix: str,
    weight_file: str = None,
    freq_file: str = None,
    scale_factor: float = None,
    freq_col: str = "FREQ",
    index_col: str = "snp",
    rg_str: str = "rg",
):
    """Summarize the results of genetic correlation analysis.

    Parameters
    ----------
    est_dir : str
        Estimation directory, containing rho<rho>.hsq, rho<rho>.log
    out_prefix : str
        output prefix
    weight_file: str
        weight_file specifying the prior variance file (<grm_prefix>.weight.tsv),
    freq_file: str
        frequency file (dataset *.snp_info files)
    scale_factor: float
        rather calculating the scale factor from `weight_file` and `freq_file` from
        scratch, specify the scale factor. This scale factor be pre-computed from
        admix.tools.gcta.calculate_hsq_scale
    freq_col: str
        column name for frequency in freq_file
    index_col: str
        column name for index in freq_file
    rg_str : str
        string name for rg, by default "rg", or "rho" (for legacy)

    Returns
    -------
    Log-likelihood curve for different rho: <out_prefix>.loglkl.txt
    Summarization file: <out_prefix>.summary.json. This file contains
        - poterior mode
        - highest posterior density interval (50% / 95%)
        - heritability (if grm_prefix is provided)

    Notes
    -----
    If `weight_file` and `freq_file` are provided, heritability at rg = 1
    (using rho100.hsq) will be estimated.
    """

    log_params("summarize-genet-cor", locals())

    rg_list = np.array(
        sorted(
            [
                int(os.path.basename(p).split(".")[0][len(rg_str) :])
                for p in glob.glob(os.path.join(est_dir, f"{rg_str}*.hsq"))
            ]
        )
    )
    admix.logger.info(f"rg={rg_list}")
    # read log-likelihood curve
    n_indiv = None
    loglkl_list = []
    for rg in rg_list:
        dict_reml = admix.tools.gcta.read_reml(os.path.join(est_dir, f"{rg_str}{rg}"))
        if n_indiv is None:
            n_indiv = dict_reml["n"]
        else:
            assert (
                n_indiv == dict_reml["n"]
            ), f"n_indiv={dict_reml['n']} from r={rg} different from previous one {n_indiv}"
        loglkl_list.append(dict_reml["loglik"])

    # interpolate
    rg_list = rg_list / 100

    # write raw estimation file
    pd.DataFrame({"rg": rg_list, "loglkl": loglkl_list}).to_csv(
        out_prefix + ".loglkl.txt", sep="\t", index=False
    )
    admix.logger.info(f"Log-likehood curves written to {out_prefix}.loglkl.txt")

    # summarize results
    assert rg_list[-1] == 1, "r=1 (r.hsq) should be included"
    dense_rg_list = np.linspace(min(rg_list), max(rg_list), 1001)
    dense_loglkl_list = CubicSpline(rg_list, loglkl_list)(dense_rg_list)

    dict_summary = {
        "n": n_indiv,
        "rg_mode": dense_rg_list[dense_loglkl_list.argmax()],
        "rg_hpdi(50%)": admix.data.hdi(dense_rg_list, dense_loglkl_list, ci=0.5),
        "rg_hpdi(95%)": admix.data.hdi(dense_rg_list, dense_loglkl_list, ci=0.95),
        "rg=1_pval": stats.chi2.sf((dense_loglkl_list.max() - dense_loglkl_list[-1]) * 2, df=1),
    }

    assert (weight_file is None) == (
        freq_file is None
    ), "weight_file and freq_file should be provided together"

    if (weight_file is not None) and (freq_file is not None):
        scale_factor = admix.tools.gcta.calculate_hsq_scale(
            weight_file=weight_file,
            freq_file=freq_file,
            freq_col=freq_col,
            index_col=index_col,
        )
        admix.logger.info(f"Computed hsq scale factor = {scale_factor:.3g}")

    if scale_factor is not None:
        dict_reml = admix.tools.gcta.read_reml(os.path.join(est_dir, f"{rg_str}100"))
        est_hsq, est_hsq_var = admix.tools.gcta.estimate_hsq(dict_reml, scale_factor=scale_factor)
        est_hsq_stderr = np.sqrt(est_hsq_var)
        dict_summary["hsq_est"] = est_hsq
        dict_summary["hsq_stderr"] = est_hsq_stderr

    # write summary
    with open(out_prefix + ".summary.json", "w") as f:
        json.dump(dict_summary, f, indent=4)
    admix.logger.info(f"Summary written to {out_prefix}.summary.json")


def meta_analyze_genet_cor(loglkl_files):
    """Meta-analyze the results of genetic correlation analysis.

    Parameters
    ----------
    loglkl_files : str
        file patterns of log-likelihood curve files
    """

    loglkl_files = glob.glob(loglkl_files)

    rg_list = None
    total_dense_loglik: np.ndarray = 0
    total_n = 0
    for f in loglkl_files:
        df_loglkl = pd.read_csv(f, sep="\t")
        if rg_list is None:
            rg_list = df_loglkl["rg"].values
            dense_rg_list = np.linspace(min(rg_list), max(rg_list), 1001)
        else:
            assert np.all(rg_list == df_loglkl["rg"].values)

        total_dense_loglik += CubicSpline(rg_list, df_loglkl["loglkl"].values)(dense_rg_list)
        # load f.replace(".loglkl.txt", ".summary.json")
        total_n += json.load(open(f.replace(".loglkl.txt", ".summary.json")))["n"]

    rg_mode = dense_rg_list[total_dense_loglik.argmax()]

    pval_rg_1 = stats.chi2.sf((total_dense_loglik.max() - total_dense_loglik[-1]) * 2, df=1)

    print(f"Meta-analysis results across {len(loglkl_files)} files")
    print("-" * 37)
    print(f"rg mode  = {rg_mode:4g}")
    for ci in [0.5, 0.95]:
        rg_hpdi = admix.data.hdi(dense_rg_list, total_dense_loglik, ci=ci)
        if isinstance(rg_hpdi, List):
            intervals = [f"[{i[0]:.4g}, {i[1]:.4g}]" for i in rg_hpdi]
            admix.logger.warning(
                f"Multiple intervals for ci={ci}, indicating some issues of model fit."
                " Please inspect log-liklihood curves for each trait."
            )
            print(f"{ci * 100:g}% HPDI = {' '.join(intervals)}")
        else:
            assert len(rg_hpdi) == 2
            print(f"{ci * 100:g}% HPDI = [{rg_hpdi[0]:.4g}, {rg_hpdi[1]:.4g}]")
    print(f"Null (rg = 1) p-value: {pval_rg_1:.4g}")
    print(f"Average N={int(np.round(total_n / len(loglkl_files)))}")


def admix_grm_rho(prefix: str, out_dir: str, rho_list=np.linspace(0, 1.0, 21)) -> None:
    """
    DEPRECATED. Will be removed in future versions.

    Build the GRM for a given rho list

    Parameters
    ----------
    prefix : str
        Prefix of the GRM files, with .K1.grm.bin and .K2.grm.bin
    out_dir : str
        folder to store the output files
    rho_list : list, optional
        List of rho values, by default np.linspace(0, 1.0, 21)
    """
    log_params("admix-grm-rho", locals())

    K1, df_id1, n_snps1 = admix.tools.gcta.read_grm(prefix + ".K1")
    K2, df_id2, n_snps2 = admix.tools.gcta.read_grm(prefix + ".K2")
    assert df_id1.equals(df_id2)
    assert np.allclose(n_snps1, n_snps2)
    df_id = df_id1
    n_snps = n_snps1

    os.makedirs(out_dir, exist_ok=True)
    for rho in tqdm(rho_list):
        K = K1 + K2 * rho

        name = os.path.join(out_dir, f"rho{int(rho * 100)}")
        admix.tools.gcta.write_grm(
            name,
            K=K,
            df_id=df_id,
            n_snps=n_snps,
        )


def estimate_genetic_cor(
    pheno: str,
    out_dir: str,
    grm_dir: str = None,
    grm_prefix: str = None,
    quantile_normalize: bool = True,
    n_thread: int = 2,
):
    """
    DEPRECATED. Will be removed in future versions.
    Estimate genetic correlation from a set of GRM files (with different rho values)

    Parameters
    ----------
    pheno : str
        phenotype file, the 1st column contains ID, 2nd column contains phenotype, and
        the rest of columns are covariates.
    out_dir : str
        folder to store the output files
    grm_dir : str
        folder containing GRM files

    quantile_normalize: bool
        whether to perform quantile normalization for both phenotype and each column of covariates
    n_thread : int, optional
        number of threads, by default 2
    """
    log_params("estimate-genetic-cor", locals())

    # either grm_dir or grm_prefix must be specified
    assert (grm_dir is not None) + (
        grm_prefix is not None
    ) == 1, "Either grm_dir or grm_prefix must be specified"

    # compile phenotype and covariates
    df_pheno = pd.read_csv(pheno, sep="\t", index_col=0)
    df_pheno.index = df_pheno.index.astype(str)

    # subset for individuals with non-nan value in df_trait
    trait_col = df_pheno.columns[0]
    covar_cols = df_pheno.columns[1:]

    # filter out individuals with missing phenotype
    df_pheno = df_pheno[df_pheno[trait_col].notna()]

    df_trait = df_pheno[[trait_col]].copy()
    df_covar = df_pheno[covar_cols].copy()
    df_covar = admix.data.convert_dummy(df_covar)
    if quantile_normalize:
        # perform quantile normalization
        for col in df_trait.columns:
            df_trait[col] = admix.data.quantile_normalize(df_trait[col])

        for col in df_covar.columns:
            df_covar[col] = admix.data.quantile_normalize(df_covar[col])

    # fill na with column mean
    df_covar.fillna(df_covar.mean(), inplace=True)

    df_id = pd.DataFrame(
        {"FID": df_trait.index.values, "IID": df_trait.index.values},
        index=df_trait.index.values,
    )
    df_trait = pd.merge(df_id, df_trait, left_index=True, right_index=True)
    df_covar = pd.merge(df_id, df_covar, left_index=True, right_index=True)

    os.makedirs(out_dir, exist_ok=True)

    if grm_dir is not None:
        # fit different rho
        grm_prefix_list = [
            p.split("/")[-1][: -len(".grm.bin")]
            for p in glob.glob(os.path.join(grm_dir, "*.grm.bin"))
        ]
    else:
        assert grm_prefix is not None
        grm_dir = os.path.dirname(grm_prefix)
        grm_prefix_list = [grm_prefix.split("/")[-1]]

    for grm_prefix in grm_prefix_list:
        grm = os.path.join(grm_dir, grm_prefix)
        out_prefix = os.path.join(out_dir, grm_prefix)
        if not os.path.exists(out_prefix + ".hsq"):
            admix.tools.gcta.reml(
                grm_path=grm,
                df_pheno=df_trait,
                df_covar=df_covar,
                out_prefix=out_prefix,
                n_thread=n_thread,
                est_fix=True,
            )

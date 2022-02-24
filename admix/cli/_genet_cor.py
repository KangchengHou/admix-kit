import admix
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
import glob
from natsort import natsorted
import os
from ._utils import log_params


def _write_admix_grm(
    K1: np.ndarray,
    K2: np.ndarray,
    df_weight: pd.Series,
    df_id: pd.DataFrame,
    out_prefix: str,
):
    """write admix grm

    Parameters
    ----------
    K1 : np.ndarray
        K1 matrix
    K2 : np.ndarray
        K2 matrix
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
    for i, K in enumerate([K1, K2]):
        name = f".K{i+1}"
        admix.tools.gcta.write_grm(
            out_prefix + name,
            K=K,
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

    Returns
    -------
    GRM files: {out_prefix}.[K1, K2].[grm.bin | grm.id | grm.n] will be generated
    Weight file: {out_prefix}.weight.tsv will be generated
    """

    log_params("admix-grm", locals())
    assert len(freq_cols) == 2, "freq_cols must be a list of length 2"
    dset = admix.io.read_dataset(pfile=pfile, snp_chunk=512)
    assert dset.n_anc == 2, "Currently only 2-way admixture is supported"

    snp_subset = np.where(
        dset.snp[freq_cols[0]].between(maf_cutoff, 1 - maf_cutoff)
        & dset.snp[freq_cols[1]].between(maf_cutoff, 1 - maf_cutoff)
    )[0]

    dset = dset[snp_subset]
    dset.snp["PRIOR_VAR"] = admix.data.calc_snp_prior_var(dset.snp, her_model=her_model)

    G1, G2, G12 = admix.data.admix_grm(
        geno=dset.geno,
        lanc=dset.lanc,
        snp_prior_var=dset.snp.PRIOR_VAR.values,
    )

    K1 = G1 + G2
    K2 = G12 + G12.T

    _write_admix_grm(
        K1=K1,
        K2=K2,
        df_weight=dset.snp.PRIOR_VAR,
        df_id=pd.DataFrame(
            {"0": dset.indiv.index.values, "1": dset.indiv.index.values}
        ),
        out_prefix=out_prefix,
    )


def admix_grm_merge(prefix: str, out_prefix: str, n_part: int = 22) -> None:
    """
    Merge multiple GRM matrices

    Parameters
    ----------
    prefix : str
        Prefix of the GRM files, any files with the pattern of <prefix>.<suffix>.K1.grm.bin
        will be merged
    out_prefix : str
        Prefix of the output file
    n_part : int, optional
        Number of partitions, by default 22

    Compute the GRM and store to {prefix}[.A1.npy | .A2.npy | .weight.tsv]
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
    admix.logger.info(
        f"{len(grm_prefix_list)} GRM files to be merged: {grm_prefix_list}"
    )
    prior_var_list = []
    for grm_prefix in grm_prefix_list:
        prior_var_list.append(
            pd.read_csv(grm_prefix + ".weight.tsv", sep="\t", index_col=0)
        )

    def _merge(suffix):
        total_grm = 0
        total_df_id = None

        weight_list = []
        for i, grm_prefix in enumerate(grm_prefix_list):
            prior_var = prior_var_list[i]
            weight = prior_var["PRIOR_VAR"].sum()
            weight_list.append(weight)
            grm, df_id, n_snps = admix.tools.gcta.read_grm(f"{grm_prefix}.{suffix}")
            total_grm += grm * weight
            if total_df_id is None:
                total_df_id = df_id
            else:
                assert np.all(total_df_id == df_id)
            assert np.all(n_snps == prior_var.shape[0])
        total_grm /= np.sum(weight_list)
        return total_grm, df_id

    K1, df_id1 = _merge("K1")
    K2, df_id2 = _merge("K2")
    assert df_id1.equals(df_id2)
    df_id = df_id1
    df_weight = pd.concat(prior_var_list)

    _write_admix_grm(
        K1=K1,
        K2=K2,
        df_weight=df_weight,
        df_id=df_id,
        out_prefix=out_prefix,
    )


def admix_grm_rho(prefix: str, out_dir: str, rho_list=np.linspace(0, 1.0, 21)) -> None:
    """
    Build the GRM for a given rho list

    Parameters
    ----------
    prefix : str
        Prefix of the GRM files, with .K1.grm.bin and .K2.grm.bin
    out_dir : str
        folder to store the output files, must not exist
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

    os.makedirs(out_dir, exist_ok=False)
    for rho in tqdm(rho_list):
        K = K1 + K2 * rho

        name = os.path.join(out_dir, f"rho{int(rho * 100)}")
        admix.tools.gcta.write_grm(
            name,
            K=K,
            df_id=df_id,
            n_snps=n_snps,
        )

import admix
import numpy as np
from typing import List
import dapgen
import pandas as pd
import glob
import os
from ._utils import log_params


def prune(pfile: str, out: str, indep_pairwise_params: List = None):
    """Prune a pfile based on indep_pairwise_params

    Parameters
    ----------
    pfile : str
        pfile
    out : str
        out_prefix
    indep_pairwise_params : [type], optional
        if None, use the default [100 5 0.1]

    Returns
    -------
    out.[pgen|pvar|psam] will be created
    """
    log_params("prune", locals())

    if indep_pairwise_params is None:
        indep_pairwise_params = [100, 5, 0.1]

    admix.tools.plink2.prune(
        pfile=pfile,
        out_prefix=out,
        indep_pairwise_params=indep_pairwise_params,
    )


def pca(pfile: str, out: str, approx=False):
    """
    Perform PCA on a pgen file

    Parameters
    ----------
    pfile : str
        Path to the pgen file
    out : str
        Path to the output file
    approx : bool, optional
        If True, use the approximate algorithm.
    """
    log_params("pca", locals())

    admix.tools.plink2.pca(pfile=pfile, out_prefix=out, approx=approx)


def liftover(pfile: str, out: str, chain="hg19->hg38"):
    """
    Lift over a pgen file

    Parameters
    ----------
    pfile : str
        Path to the pgen file
    out : str
        Path to the output file
    chain : str, optional
        Chain file to use.
    """
    log_params("liftover", locals())

    admix.tools.plink2.lift_over(pfile=pfile, out_prefix=out, chain=chain)


def pfile_merge_indiv(pfile1: str, pfile2: str, out: str):
    log_params("pfile-merge-indiv", locals())
    admix.tools.plink2.merge_indiv(pfile1=pfile1, pfile2=pfile2, out_prefix=out)
    admix.logger.info(
        f"Currently PLINK2 pmerge does not fully support merging pfiles with different individuals, writing PLINK1 bed file instead."
    )
    admix.logger.info(f"{out}.bed, {out}.bim, {out}.fam are created")


def pfile_align_snp(pfile1: str, pfile2: str, out: str):
    """align the SNP for 2 pfiles

    Parameters
    ----------
    pfile1 : str
        plink file 1
    pfile2 : str
        plink file 2
    out : str
        output prefix
    """
    log_params("pfile-align-snp", locals())
    admix.tools.plink2.align_snp(pfile1=pfile1, pfile2=pfile2, out_prefix=out)


def pfile_freq_within(
    pfile: str,
    group_col: str,
    out: str,
    groups: str = None,
):
    """calculate the within-cluster frequencies for a pfile using
    plink --freq --within

    Parameters
    ----------
    pfile : str
        plink file
    col : str
        column
    out : str
        output prefix
    groups : str, optional
        unique groups
    """
    log_params("pfile-freq-within", locals())

    # 1. convert pfile into bfile
    bfile = f"{out}-tmp"
    admix.tools.plink2.run(f"--pfile {pfile} --make-bed --out {bfile}")

    # 2. create --within file
    # https://www.cog-genomics.org/plink/1.9/input#within
    within_file = f"{out}-tmp.within"
    df_psam = dapgen.read_psam(pfile + ".psam")
    assert (
        group_col in df_psam.columns
    ), f"group-col={group_col} not in {pfile}.psam: {df_psam.columns}"
    if groups is None:
        groups = df_psam[group_col].unique()
    df_within = df_psam[df_psam[group_col].isin(groups)].copy()

    # when converting PLINK2 to PLINK1, all #FID is converted to 0
    df_within = pd.DataFrame(
        {
            "FID": ["0"] * len(df_within),
            "IID": df_within.index,
            group_col: df_within[group_col].values,
        }
    )
    df_within.to_csv(within_file, sep="\t", index=False, header=False)

    print(df_within)
    # 3. run plink --freq --within
    admix.tools.plink.run(f"--bfile {bfile} --freq --within {within_file} --out {out}")

    # 4. clean up
    for f in glob.glob(f"{out}-tmp*"):
        os.remove(f)


def subset_hapmap3(pfile: str, out: str, build: str):
    log_params("subset-hapmap3", locals())
    admix.tools.plink2.subset_hapmap3(pfile, out_prefix=out, build=build)
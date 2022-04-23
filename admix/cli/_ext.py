import admix
import numpy as np
from typing import List
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


def merge_pfile_indiv(pfile1: str, pfile2: str, out: str):
    log_params("merge-pfile-indiv", locals())
    admix.tools.plink2.merge_indiv(pfile1=pfile1, pfile2=pfile2, out_prefix=out)
    admix.logger.info(
        f"Currently PLINK2 pmerge does not fully support merging pfiles with different individuals, writing PLINK1 bed file instead."
    )
    admix.logger.info(f"{out}.bed, {out}.bim, {out}.fam are created")


def subset_hapmap3(pfile: str, out: str, build: str):
    log_params("subset-hapmap3", locals())
    admix.tools.plink2.subset_hapmap3(pfile, out_prefix=out, build=build)
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
    prune : bool
        Whether to prune the pfile using the default recipe
        --indep 200 5 1.15, --indep-pairwise 100 5 0.1
    out : str
        Path to the output file
    """
    log_params("pca", locals())

    admix.tools.plink2.pca(pfile=pfile, out_prefix=out, approx=approx)

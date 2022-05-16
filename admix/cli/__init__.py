#!/usr/bin/env python

import fire
from ._assoc import assoc
from ._geno import append_snp_info
from ._utils import log_params
from ._simulate import simulate_pheno
from ._lanc import lanc, lanc_convert, lanc_count
from ._ext import (
    prune,
    pca,
    liftover,
    subset_hapmap3,
    pfile_align_snp,
    pfile_merge_indiv,
    pfile_freq_within,
)
from ._plot import plot_pca
from ._genet_cor import admix_grm, admix_grm_merge, admix_grm_rho, estimate_genetic_cor


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()


if __name__ == "__main__":
    fire.Fire()
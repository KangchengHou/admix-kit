#!/usr/bin/env python

import fire
from ._assoc import assoc
from ._geno import append_snp_info, calc_pgs, calc_partial_pgs, grm
from ._utils import log_params, get_1kg_ref, select_admix_indiv
from ._simulate import simulate_admix_pheno, simulate_pheno
from ._lanc import lanc, lanc_convert, lanc_count
from ._ext import (
    prune,
    pca,
    liftover,
    pfile_align_snp,
    pfile_merge_indiv,
    pfile_freq_within,
    subset_hapmap3,
    subset_pop_indiv,
    hapgen2,
    admix_simu,
    haptools_simu_admix,
    download_dependency,
)
from ._plot import plot_joint_pca
from ._genet_cor import (
    admix_grm,
    admix_grm_merge,
    genet_cor,
    admix_grm_rho,
    estimate_genetic_cor,
    summarize_genet_cor,
    meta_analyze_genet_cor,
)


def cli():
    """
    Entry point for the admix command line interface.
    """
    fire.Fire()


if __name__ == "__main__":
    fire.Fire()

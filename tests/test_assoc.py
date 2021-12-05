import numpy as np
import pandas as pd
import admix


def test_consistent():
    """
    Test the consistency of the association between the precomputed file and new file
    """
    dset = admix.dataset.load_toy_admix()

    # how two implementations cope with zero-freq SNPs are a bit different for now.
    # subset the dataset to avoid this for now
    test_eq_idx = (
        ~((dset.snp.FREQ1.isin([0, 1])) | (dset.snp.FREQ2.isin([0, 1])))
    ).values
    dset = dset[test_eq_idx]

    pval_att = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="ATT", fast=False
    )
    pval_tractor = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="TRACTOR", fast=False
    )
    assert np.allclose(pval_att, dset.snp["ATT"])
    assert np.allclose(pval_tractor, dset.snp["TRACTOR"])

    pval_att_fast = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="ATT", fast=True
    )
    assert np.allclose(pval_att_fast, pval_att)

    # figure out inconsistency between fast and slow
    pval_tractor_fast = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="TRACTOR", fast=True
    )
    assert np.allclose(pval_tractor_fast, pval_tractor)

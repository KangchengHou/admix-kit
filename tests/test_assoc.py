import numpy as np
import pandas as pd
import admix


def test_consistent():
    """
    Test the consistency of the association between the precomputed file and new file
    """
    dset = admix.dataset.load_toy_admix()
    pval_att = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="ATT", fast=False
    )
    pval_tractor = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="TRACTOR", fast=False
    )
    assert np.allclose(pval_att, dset.snp["ATT"], equal_nan=True)
    assert np.allclose(pval_tractor, dset.snp["TRACTOR"], equal_nan=True)

    pval_att_fast = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="ATT", fast=True
    )
    # TODO: test all values
    assert np.allclose(pval_att_fast[0:5], pval_att[0:5], equal_nan=True)

    # figure out inconsistency between fast and slow
    pval_tractor_fast = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="TRACTOR", fast=True
    )

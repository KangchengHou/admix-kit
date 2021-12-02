import numpy as np
import pandas as pd
import admix


def test_consistent():
    """
    Test the consistency of the association between the precomputed file and new file
    """
    dset = admix.dataset.load_toy_admix()
    pval_att = admix.assoc.marginal(dset, pheno=dset.indiv.PHENO, method="ATT")
    pval_tractor = admix.assoc.marginal(dset, pheno=dset.indiv.PHENO, method="TRACTOR")
    assert np.allclose(pval_att, dset.snp["ATT"], equal_nan=True)
    assert np.allclose(pval_tractor, dset.snp["TRACTOR"], equal_nan=True)

    # TODO: add test also for marginal_fast for the consistency of the results
    # pval_tractor = admix.assoc.marginal_fast(dset, pheno=dset.indiv.PHENO, method="ATT")
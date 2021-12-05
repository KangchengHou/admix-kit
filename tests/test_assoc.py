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
        (dset.snp.FREQ1.between(0.01, 0.99)) & (dset.snp.FREQ2.between(0.01, 0.99))
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

    pval_tractor_fast = admix.assoc.marginal(
        dset, pheno=dset.indiv.PHENO, method="TRACTOR", fast=True
    )
    assert np.allclose(pval_tractor_fast, pval_tractor)


def test_linear_nan():
    """
    Test the consistency of the linear association between the precomputed file and new file
    in the presence of NaNs
    """
    dset = admix.dataset.load_toy_admix()

    # how two implementations cope with zero-freq SNPs are a bit different for now.
    # subset the dataset to avoid this for now
    test_eq_idx = (
        dset.snp.FREQ1.between(0.05, 0.95) & (dset.snp.FREQ2.between(0.05, 0.95))
    ).values
    dset = dset[test_eq_idx]

    # obtain geno and lanc
    geno = dset.geno.compute()
    lanc = dset.lanc.compute()
    ref_pval_att = admix.assoc.marginal(
        geno=geno, lanc=lanc, pheno=dset.indiv.PHENO, method="ATT", fast=True
    )
    ref_pval_tractor = admix.assoc.marginal(
        geno=geno, lanc=lanc, pheno=dset.indiv.PHENO, method="TRACTOR", fast=True
    )
    assert np.allclose(ref_pval_att, dset.snp["ATT"])
    assert np.allclose(ref_pval_tractor, dset.snp["TRACTOR"])

    # set some of the 3rd SNP of geno to NaN
    geno_with_nan = geno.copy()
    geno_with_nan[3, 0, 1] = np.NaN
    geno_with_nan[3, 2, 0] = np.NaN
    geno_with_nan[5, 0, 0] = np.NaN
    geno_with_nan[5, 2, 0] = np.NaN
    pval_att_fast = admix.assoc.marginal(
        geno=geno_with_nan, lanc=lanc, pheno=dset.indiv.PHENO, method="ATT", fast=True
    )
    pval_tractor_fast = admix.assoc.marginal(
        geno=geno_with_nan,
        lanc=lanc,
        pheno=dset.indiv.PHENO,
        method="TRACTOR",
        fast=True,
    )
    # all SNPs except the  should be the same as the reference
    diff_idx = [3, 5]
    same_idx = np.delete(np.arange(dset.n_snp), diff_idx)
    assert np.allclose(pval_att_fast[same_idx], ref_pval_att[same_idx])
    assert np.allclose(pval_tractor_fast[same_idx], ref_pval_tractor[same_idx])
    # the NaN SNPs should be different
    assert np.all(pval_att_fast[diff_idx] != ref_pval_att[diff_idx])
    assert np.all(pval_tractor_fast[diff_idx] != ref_pval_tractor[diff_idx])

    # assess consistency between fast and slow
    # randomly add nan to the geno
    for _ in range(5):
        geno_with_nan = geno.copy()
        geno_with_nan[
            np.random.choice(dset.n_snp, size=2, replace=False),
            np.random.choice(dset.n_indiv, size=2, replace=False),
            0,
        ] = np.NaN
        geno_with_nan[
            np.random.choice(dset.n_snp, size=2, replace=False),
            np.random.choice(dset.n_indiv, size=2, replace=False),
            1,
        ] = np.NaN

        pval_tractor_slow = admix.assoc.marginal(
            geno=geno_with_nan,
            lanc=lanc,
            pheno=dset.indiv.PHENO,
            method="TRACTOR",
            fast=False,
        )
        pval_tractor_fast = admix.assoc.marginal(
            geno=geno_with_nan,
            lanc=lanc,
            pheno=dset.indiv.PHENO,
            method="TRACTOR",
            fast=True,
        )
        assert np.allclose(pval_tractor_slow, pval_tractor_fast)
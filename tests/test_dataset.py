import dask.array as da
import numpy as np
import admix


def test_dataset():
    dset = admix.dataset.load_toy_admix()
    lanc = dset.lanc
    geno = dset.geno
    pos = dset.snp.POS

    # basic shape
    assert lanc.ndim == 3
    assert geno.ndim == 3
    assert np.all(lanc.shape == geno.shape)
    assert len(pos) == lanc.shape[0]

    # subset data set
    snp_idx = [100, 200, 300]
    indiv_idx = [1, 3, 5, 7]
    dset_subset = dset[snp_idx, indiv_idx]
    assert dset.snp.iloc[snp_idx].equals(dset_subset.snp)
    assert dset.indiv.iloc[indiv_idx].equals(dset_subset.indiv)
    assert np.all(dset.geno.compute()[np.ix_(snp_idx, indiv_idx)] == dset_subset.geno)
    assert np.all(dset.lanc.compute()[np.ix_(snp_idx, indiv_idx)] == dset_subset.lanc)

    # subset with snp and indiv names
    dset_subset = dset[
        dset.snp.index[snp_idx].values, dset.indiv.index[indiv_idx].values
    ]
    assert dset.snp.iloc[snp_idx].equals(dset_subset.snp)
    assert dset.indiv.iloc[indiv_idx].equals(dset_subset.indiv)
    assert np.all(dset.geno.compute()[np.ix_(snp_idx, indiv_idx)] == dset_subset.geno)
    assert np.all(dset.lanc.compute()[np.ix_(snp_idx, indiv_idx)] == dset_subset.lanc)

import dask.array as da
import numpy as np
import pandas as pd
import admix
import os
import zarr


def get_data_path(fn):
    return os.path.join(os.path.dirname(__file__), 'test-data', fn)


def test_basic():
    a = zarr.load(get_data_path("admix.zarr"))
    lanc = a["lanc"]
    hap = a["hap"]
    pos = np.loadtxt(get_data_path("pos.txt"), dtype=int)

    # basic shape
    assert lanc.ndim == 3
    assert hap.ndim == 3
    assert np.all(lanc.shape == hap.shape)
    assert len(pos) == lanc.shape[1]

    # # phenotype simulation
    # sim_pheno = admix.simulate.simulate_phenotype_case_control_1snp(
    #     hap=hap, lanc=lanc, case_prevalence=0.1, odds_ratio=1.0, n_sim=10
    # )
    # assert len(sim_pheno) == n_snp
    # assert np.all(sim_pheno[0].shape == (n_hap // 2, 10))


def test_utils():
    # test convert anc count
    lanc = np.array([[[0, 0], [0, 0], [0, 0]]])
    hap = np.array([[[0, 1], [1, 1], [0, 0]]])
    allele_per_anc = admix.data.compute_allele_per_anc(hap=hap, lanc=lanc, n_anc=2).compute()
    assert np.all(allele_per_anc == [[[1, 0], [2, 0], [0, 0]]])

    path_zarr = get_data_path("admix.zarr")
    lanc = da.from_zarr(path_zarr, "lanc")
    hap = da.from_zarr(path_zarr, "hap")
    allele_per_anc = admix.data.compute_allele_per_anc(hap=hap, lanc=lanc, n_anc=2)
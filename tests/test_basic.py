import dask.array as da
import numpy as np
import pandas as pd
import admix
import os
import zarr


def get_data_path(fn):
    return os.path.join("./tests/test-data", fn)
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
    # tests convert anc count
    lanc = np.array([[[0, 0], [0, 0], [0, 0]]])
    hap = np.array([[[0, 1], [1, 1], [0, 0]]])
    allele_per_anc = admix.data.compute_allele_per_anc(hap=hap, lanc=lanc, n_anc=2).compute()
    assert np.all(allele_per_anc == [[[1, 0], [2, 0], [0, 0]]])

    path_zarr = get_data_path("admix.zarr")
    lanc = da.from_zarr(path_zarr, "lanc")
    hap = da.from_zarr(path_zarr, "hap")
    allele_per_anc = admix.data.compute_allele_per_anc(hap=hap, lanc=lanc, n_anc=2)

def test_compute_grm():
    path_zarr = get_data_path("admix.zarr")
    lanc = da.from_zarr(path_zarr, "lanc")
    hap = da.from_zarr(path_zarr, "hap")
    allele_per_anc = admix.data.compute_allele_per_anc(hap=hap, lanc=lanc, n_anc=2).astype(float)

def test_simulate():
    from admix.simulate import simulate_continuous_phenotype, simulate_continuous_phenotype_grm
    from admix.data import compute_admix_grm
    path_zarr = get_data_path("admix.zarr")
    lanc = da.from_zarr(path_zarr, "lanc")
    hap = da.from_zarr(path_zarr, "hap")
    beta, phe_g, phe = simulate_continuous_phenotype(hap=hap, lanc=lanc, var_g=1.0, gamma=1.0, var_e=1.0)

    K1, K2 = compute_admix_grm(hap, lanc, n_anc=2)
    ys = simulate_continuous_phenotype_grm(K1=K1, K2=K2, var_g=1.0, gamma=1.0, var_e=1.0)

def test_lamp():
    from pylampld import LampLD
    admix_lanc = da.from_zarr(get_data_path("admix.zarr"), "lanc")
    admix_hap = da.from_zarr(get_data_path("admix.zarr"), "hap")

    eur_hap = da.from_zarr(get_data_path("eur.zarr")).compute()
    afr_hap = da.from_zarr(get_data_path("afr.zarr")).compute()
    eur_hap = np.vstack([eur_hap[:, :, 0], eur_hap[:, :, 1]])
    afr_hap = np.vstack([afr_hap[:, :, 0], afr_hap[:, :, 1]])
    ref_list = [eur_hap, afr_hap]
    n_anc = len(ref_list)
    n_snp = admix_hap.shape[1]
    model = LampLD(
        n_snp=n_snp,
        n_anc=n_anc,
        n_proto=6,
        window_size=300
    )
    pos = np.loadtxt(get_data_path("pos.txt"), dtype=int)
    model.set_pos(pos)
    model.fit(ref_list)

    est = np.dstack([model.infer_lanc(admix_hap[:, :, 0]),
                     model.infer_lanc(admix_hap[:, :, 1])])
    acc = (est == admix_lanc).mean().compute()
    assert acc > 0.9
    print(f"Accuracy: {acc:.2f}")


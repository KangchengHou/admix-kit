import dask.array as da
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import admix
import os
import zarr
import xarray as xr
from admix.data import load_toy


def test_basic():
    ds = load_toy()[0]
    lanc = ds["lanc"]
    geno = ds["geno"]
    pos = ds["position@snp"]

    # basic shape
    assert lanc.ndim == 3
    assert geno.ndim == 3
    assert np.all(lanc.shape == geno.shape)
    assert len(pos) == lanc.shape[1]

    # phenotype simulation
    # sim_pheno = admix.simulate.simulate_phenotype_case_control_1snp(
    #     hap=hap, lanc=lanc, case_prevalence=0.1, odds_ratio=1.0, n_sim=10
    # )
    # assert len(sim_pheno) == n_snp
    # assert np.all(sim_pheno[0].shape == (n_hap // 2, 10))


def test_utils():
    # tests convert anc count
    geno = np.array([[[0, 1], [1, 1], [0, 0]]])
    lanc = np.array([[[0, 0], [0, 0], [0, 0]]])
    ds = xr.Dataset(
        data_vars={
            "geno": (("indiv", "snp", "haploid"), geno),
            "lanc": (("indiv", "snp", "haploid"), lanc),
        },
        attrs={"n_anc": 2},
    )

    allele_per_anc = admix.data.compute_allele_per_anc(ds).compute()
    assert np.all(allele_per_anc == [[[1, 0], [2, 0], [0, 0]]])
    ds = load_toy()[0]
    allele_per_anc = admix.data.compute_allele_per_anc(ds)


def test_compute_grm():
    ds = load_toy()[0]
    allele_per_anc = admix.data.compute_allele_per_anc(ds).astype(float)


def test_simulate():
    from admix.simulate import continuous_phenotype, continuous_phenotype_grm
    from admix.data import compute_admix_grm

    ds_admix = load_toy()[0]
    beta, phe_g, phe = continuous_phenotype(ds_admix, var_g=1.0, gamma=1.0, var_e=1.0)
    K1, K2 = compute_admix_grm(ds_admix)
    ys = continuous_phenotype_grm(K1=K1, K2=K2, var_g=1.0, gamma=1.0, var_e=1.0)


def test_lamp():
    from pylampld import LampLD

    ds_admix, ds_eur, ds_afr = load_toy()
    eur_hap = np.vstack([ds_eur["geno"][:, :, 0], ds_eur["geno"][:, :, 1]])
    afr_hap = np.vstack([ds_afr["geno"][:, :, 0], ds_afr["geno"][:, :, 1]])
    ref_list = [eur_hap, afr_hap]
    n_anc = len(ref_list)
    n_snp = ds_admix.dims["snp"]
    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=6, window_size=300)
    model.set_pos(ds_admix["position@snp"].values)
    model.fit(ref_list)

    est = np.dstack(
        [
            model.infer_lanc(ds_admix["geno"][:, :, 0].compute()),
            model.infer_lanc(ds_admix["geno"][:, :, 1].compute()),
        ]
    )

    acc = (est == ds_admix["lanc"].values).mean()
    assert acc > 0.9
    print(f"Accuracy: {acc:.2f}")

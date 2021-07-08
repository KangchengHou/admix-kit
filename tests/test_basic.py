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

    allele_per_anc = admix.tools.allele_per_anc(ds, inplace=False)
    assert np.all(allele_per_anc == [[[1, 0], [2, 0], [0, 0]]])
    ds = load_toy()[0]
    apa1 = admix.tools.allele_per_anc(ds, return_mask=False, inplace=False)
    apa2 = admix.tools.allele_per_anc(ds, return_mask=True, inplace=False)
    assert np.all(apa1 == np.ma.getdata(apa2)).compute()
    assert np.all(apa1.compute()[da.ma.getmaskarray(apa2)] == 0)


def test_compute_grm():

    ds = load_toy()[0]
    allele_per_anc = admix.tools.allele_per_anc(ds, inplace=False).astype(float)


def test_simulate():
    from admix.simulate import continuous_pheno, continuous_pheno_grm
    from admix.tools import admix_grm

    dset = load_toy()[0]
    sim = continuous_pheno(dset, var_g=1.0, gamma=1.0, var_e=1.0)
    grm = admix_grm(dset, inplace=False)
    ys = continuous_pheno_grm(dset, grm, var_g=1.0, gamma=1.0, var_e=1.0)


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


def test_assoc():
    """
    TODO: add basic testing to association testing modules.
    """
    from admix.simulate import continuous_pheno

    admix_dset, eur_dset, afr_dset = admix.data.load_toy()
    sim = continuous_pheno(admix_dset, var_g=1.0, gamma=1.0, var_e=1.0)
    i_sim = 0
    sim_beta = sim["beta"][:, :, i_sim]
    sim_pheno = sim["pheno"][:, i_sim]

    assoc = admix.assoc.marginal(
        dset=admix_dset.assign_coords(pheno=("indiv", sim_pheno)),
        pheno="pheno",
        method="ATT",
        family="linear",
    )
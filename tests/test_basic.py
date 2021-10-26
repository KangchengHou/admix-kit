import dask.array as da
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
import admix

import xarray as xr
from admix.data import load_toy


def test_basic():
    ds = load_toy()[0]
    lanc = ds["lanc"]
    geno = ds["geno"]
    pos = ds["POS"]

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
            "geno": (("indiv", "snp", "ploidy"), da.from_array(geno)),
            "lanc": (("indiv", "snp", "ploidy"), da.from_array(lanc)),
        },
        attrs={"n_anc": 2},
    )

    allele_per_anc = admix.tools.allele_per_anc(ds, inplace=False)
    assert np.all(allele_per_anc == [[[1, 0], [2, 0], [0, 0]]])
    ds = load_toy()[0]
    apa = admix.tools.allele_per_anc(ds, inplace=False)


def test_compute_grm():

    ds = load_toy()[0]
    allele_per_anc = admix.tools.allele_per_anc(ds, inplace=False).astype(float)


def test_simulate():

    dset = load_toy()[0]
    sim = admix.simulate.continuous_pheno(dset, var_g=1.0, gamma=1.0, var_e=1.0)

    admix.tools.grm(dset, method="center")
    ys = admix.simulate.continuous_pheno_grm(
        dset, grm="grm", var={"grm": 1.0, "e": 1.0}
    )


def test_lamp():
    from pylampld import LampLD

    ds_admix, ds_eur, ds_afr = load_toy()
    eur_hap = np.vstack([ds_eur["geno"][:, :, 0], ds_eur["geno"][:, :, 1]])
    afr_hap = np.vstack([ds_afr["geno"][:, :, 0], ds_afr["geno"][:, :, 1]])
    ref_list = [eur_hap, afr_hap]
    n_anc = len(ref_list)
    n_snp = ds_admix.dims["snp"]
    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=6, window_size=300)
    model.set_pos(ds_admix["POS"].values)
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

    np.random.seed(1234)

    admix_dset, eur_dset, afr_dset = admix.data.load_toy()
    sim = continuous_pheno(admix_dset, var_g=1.0, gamma=1.0, var_e=1.0)
    i_sim = 0
    sim_beta = sim["beta"][:, :, i_sim]
    sim_pheno = sim["pheno"][:, i_sim]

    assoc = admix.assoc.marginal_fast(
        dset=admix_dset.assign_coords(pheno=("indiv", sim_pheno)),
        pheno="pheno",
        method="ATT",
        family="linear",
    )


def test_consistent():
    """
    Test the consistency of results with the legacy version

    test-data/test-consistent-data.json is generated using the following code.

    import admix
    import numpy as np
    import pickle
    np.random.seed(1234)

    admix_dset, _, _ = admix.data.load_toy()
    admix_dset = admix_dset.isel(snp=np.arange(100))
    af = admix.tools.af_per_anc(admix_dset, inplace=False).compute()
    apa = admix.tools.allele_per_anc(admix_dset, inplace=False, center=True).compute()

    sim = admix.simulate.continuous_pheno(admix_dset, var_g=1.0, gamma=0.8, var_e=1.0)
    sim_i = 3
    sim_beta = sim["beta"][:, :, sim_i]
    sim_pheno = sim["pheno"][:, sim_i]

    assoc = admix.assoc.marginal_fast(
        dset=admix_dset.assign_coords(pheno=("indiv", sim_pheno)),
        pheno="pheno",
        method="ATT",
        family="linear",
    )
    data_dict = {
        "af": af,
        "apa": np.swapaxes(apa, 0, 1),
        "beta": sim_beta,
        "pheno": sim_pheno,
        "assoc": assoc.P.values,
    }
    with open("test-data/test-consistent-data.pkl", 'wb') as f:
        pickle.dump(data_dict, f)
    """

    import pickle
    from os.path import dirname, join

    np.random.seed(1234)

    admix_dset, _, _ = admix.data.load_toy()
    admix_dset = admix_dset.isel(snp=np.arange(100))
    af = admix.tools.af_per_anc(admix_dset, inplace=False).compute()
    apa = admix.tools.allele_per_anc(admix_dset, inplace=False, center=True).compute()

    sim = admix.simulate.continuous_pheno(admix_dset, var_g=1.0, gamma=0.8, var_e=1.0)
    sim_i = 3
    sim_beta = sim["beta"][:, :, sim_i]
    sim_pheno = sim["pheno"][:, sim_i]

    assoc = admix.assoc.marginal_fast(
        dset=admix_dset.assign_coords(pheno=("indiv", sim_pheno)),
        pheno="pheno",
        method="ATT",
        family="linear",
    )
    test_data_path = join(dirname(__file__), "test-data")

    with open(join(test_data_path, "test-consistent-data.pkl"), "rb") as f:
        data_dict = pickle.load(f)

    assert np.allclose(data_dict["af"], af)
    assert np.allclose(data_dict["apa"], np.swapaxes(apa, 0, 1))
    assert np.allclose(data_dict["beta"], sim_beta)
    assert np.allclose(data_dict["pheno"], sim_pheno)
    assert np.allclose(data_dict["assoc"], assoc.P.values)


def test_ext_tools():
    admix.tools.get_dependency("plink2")

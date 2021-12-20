"""
Check whether basic functions runs without error
"""
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


def test_utils():
    # tests convert anc count
    geno = np.array([[[0, 1], [1, 1], [0, 0]]])
    lanc = np.array([[[0, 0], [0, 0], [0, 0]]])

    dset = admix.Dataset(
        geno=da.from_array(np.swapaxes(geno, 0, 1)),
        lanc=da.from_array(np.swapaxes(lanc, 0, 1)),
        n_anc=2,
    )
    apa = dset.allele_per_anc()
    af = dset.af_per_anc()
    assert np.all(np.swapaxes(apa, 0, 1) == [[[1, 0], [2, 0], [0, 0]]])


# def test_simulate():

#     dset = load_toy()[0]
#     sim = admix.simulate.quant_pheno(dset, var_g=1.0, gamma=1.0, var_e=1.0)

#     admix.tools.grm(dset, method="center")
#     ys = admix.simulate.quant_pheno_grm(
#         dset, grm="grm", var={"grm": 1.0, "e": 1.0}
#     )


# def test_lamp():
#     from pylampld import LampLD

#     ds_admix, ds_eur, ds_afr = load_toy()
#     eur_hap = np.vstack([ds_eur["geno"][:, :, 0], ds_eur["geno"][:, :, 1]])
#     afr_hap = np.vstack([ds_afr["geno"][:, :, 0], ds_afr["geno"][:, :, 1]])
#     ref_list = [eur_hap, afr_hap]
#     n_anc = len(ref_list)
#     n_snp = ds_admix.dims["snp"]
#     model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=6, window_size=300)
#     model.set_pos(ds_admix["POS"].values)
#     model.fit(ref_list)

#     est = np.dstack(
#         [
#             model.infer_lanc(ds_admix["geno"][:, :, 0].compute()),
#             model.infer_lanc(ds_admix["geno"][:, :, 1].compute()),
#         ]
#     )

#     acc = (est == ds_admix["lanc"].values).mean()
#     assert acc > 0.9
#     print(f"Accuracy: {acc:.2f}")


def test_assoc():
    np.random.seed(1234)
    dset_admix = admix.dataset.load_toy_admix()
    sim = admix.simulate.quant_pheno(dset_admix, hsq=0.5, cor=1.0)
    i_sim = 0
    sim_beta = sim["beta"][:, :, i_sim]
    sim_pheno = sim["pheno"][:, i_sim]

    assoc = admix.assoc.marginal(
        dset=dset_admix,
        pheno=sim_pheno,
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

    admix_dset, _, _ = admix.dataset.load_toy()
    admix_dset = admix.dataset.subset_dataset(
        admix_dset, snp=admix_dset.snp.index.values[np.arange(100)]
    )
    admix_dset.compute_af_per_anc()
    admix_dset.compute_allele_per_anc(center=True)

    af = admix_dset.af_per_anc
    apa = admix_dset.allele_per_anc.compute()

    sim = admix.simulate.quant_pheno(admix_dset, hsq=0.5, cor=0.8)
    sim_i = 3
    sim_beta = sim["beta"][:, :, sim_i]
    sim_pheno = sim["pheno"][:, sim_i]

    admix_dset.indiv["pheno"] = sim_pheno
    assoc = admix.assoc.marginal(
        dset=admix_dset,
        pheno_col="pheno",
        method="ATT",
        family="linear",
    )

    data_dict = {
        "af": af,
        "apa": apa,
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
    dset_admix, _, _ = admix.dataset.load_toy()
    dset_admix = dset_admix[0:100]

    af = dset_admix.af_per_anc()

    sim = admix.simulate.quant_pheno(dset_admix, hsq=0.5, cor=0.8)
    sim_i = 3
    sim_beta = sim["beta"][:, :, sim_i]
    sim_pheno = sim["pheno"][:, sim_i]

    dset_admix.indiv["pheno"] = sim_pheno
    assoc = admix.assoc.marginal(
        dset=dset_admix,
        pheno=sim_pheno,
        method="ATT",
        family="linear",
    )
    test_data_path = join(dirname(__file__), "test-data")

    with open(join(test_data_path, "test-consistent-data.pkl"), "rb") as f:
        data_dict = pickle.load(f)

    assert np.allclose(data_dict["af"], af)
    assert np.allclose(data_dict["beta"], sim_beta)
    assert np.allclose(data_dict["pheno"], sim_pheno)
    assert np.allclose(data_dict["assoc"], assoc)


def test_ext_tools():
    admix.tools.get_dependency("plink2")
    admix.tools.get_dependency("plink")
    admix.tools.get_dependency("liftOver")

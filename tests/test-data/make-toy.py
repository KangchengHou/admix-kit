"""
Generating toy data

1. Make PLINK2 dataset.
2. Infer local ancestry with LAMPLD.
3. Simulate phenotypes.
4. Perform association testing.
"""
import numpy as np
import pandas as pd
import dapgen
import fire
import admix


def subset_indiv():
    df_sample = dapgen.read_psam("raw.psam")
    df_sample = df_sample[df_sample.Population.isin(["CEU", "YRI", "ASW"])]

    with open("indiv.txt", "w") as f:
        f.writelines("\n".join(df_sample.index))


def process():
    np.random.seed(1234)

    dset = admix.io.read_dataset("toy-all")
    dict_pop = {
        pop: dset.indiv[dset.indiv.Population == pop].index.values
        for pop in dset.indiv.Population.unique()
    }

    admix.tools.plink2.subset("toy-all", "toy-admix", indiv_list=dict_pop["ASW"])

    dset_admix = admix.io.read_dataset("toy-admix")
    af_per_anc = dset_admix.af_per_anc()

    dict_dset = {pop: dset[:, dict_pop[pop]] for pop in ["CEU", "YRI"]}

    assert np.all(dset_admix.indiv.index == dict_pop["ASW"])
    est_lanc = admix.ancestry.lanc(
        sample_dset=dset_admix,
        ref_dsets=[dict_dset["CEU"], dict_dset["YRI"]],
        method="lampld",
        n_proto=6,
        window_size=300,
    )
    admix.io.write_lanc("toy-admix.lanc", lanc=est_lanc)

    # pick one simulation to look at
    sim_i = 1

    beta = np.zeros((dset_admix.n_snp, 2))
    beta[0, 0] = 1.0
    beta[0, 1] = 0.8
    sim = admix.simulate.quant_pheno(dset_admix, beta=beta, hsq=0.5)

    df_snp_info = {
        "BETA1": sim["beta"][:, 0, sim_i],
        "BETA2": sim["beta"][:, 1, sim_i],
        "FREQ1": af_per_anc[:, 0],
        "FREQ2": af_per_anc[:, 1],
        "ATT": admix.assoc.marginal(
            dset_admix, pheno=sim["pheno"][:, sim_i], method="ATT", fast=False
        ),
        "TRACTOR": admix.assoc.marginal(
            dset_admix, pheno=sim["pheno"][:, sim_i], method="TRACTOR", fast=False
        ),
    }
    df_snp_info = pd.DataFrame(df_snp_info, index=dset_admix.snp.index)

    df_indiv_info = {"PHENO": sim["pheno"][:, sim_i]}
    df_indiv_info = pd.DataFrame(df_indiv_info, index=dset_admix.indiv.index)

    df_snp_info.to_csv("toy-admix.snp_info", sep="\t", float_format="%.6g", na_rep="NA")
    df_indiv_info.to_csv("toy-admix.indiv_info", sep="\t", float_format="%.6f")


if __name__ == "__main__":
    fire.Fire()
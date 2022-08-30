import admix
import pandas as pd
import numpy as np
from typing import List
import dapgen
from ._utils import log_params


def simulate_admix_pheno(
    pfile: str,
    hsq: float,
    out_prefix: str,
    cor: float = 1.0,
    family: str = "quant",
    n_causal: int = None,
    p_causal: float = None,
    case_prevalence: float = 0.5,
    seed: int = None,
    snp_effect: str = None,
    n_sim: int = 10,
):
    """
    Simulate phenotypes using a pgen file for admixed individuals.

    Parameters
    ----------
    pfile : str
        Path to the pgen file
    hsq : float
        Heritability
    out_prefix : str
        prefix to the output file, <out>.pheno, <out>.snpeffect will be created
    cor : float
        genetic correlation across local ancestries
    family : str
        phenotype type to simulate, either "quant" or "binary"
    n_causal : int
        Number of causal variants to simulate
    p_causal : float
        Proportion of a causal variant
    case_prevalence: float
        Prevalence of cases, default 0.5
    seed : int
        Random seed
    beta : str
        Path to the beta file
    n_sim : int
        Number of simulations to perform
    """
    log_params("simulate-pheno", locals())
    assert snp_effect is None, "snp_effect is not supported yet"
    if seed is not None:
        np.random.seed(seed)

    assert (n_causal is not None) + (
        p_causal is not None
    ) == 1, "one of n_causal/p_causal must be specified"

    dset = admix.io.read_dataset(pfile)

    if p_causal is not None:
        n_causal = int(dset.n_indiv * p_causal)

    if family == "quant":
        dict_sim = admix.simulate.quant_pheno(
            dset=dset,
            hsq=hsq,
            cor=cor,
            n_causal=n_causal,
            n_sim=n_sim,
            beta=snp_effect,
        )
    elif family == "binary":
        dict_sim = admix.simulate.binary_pheno(
            dset=dset,
            hsq=hsq,
            cor=cor,
            case_prevalence=case_prevalence,
            n_causal=n_causal,
            n_sim=n_sim,
            beta=snp_effect,
        )
    else:
        raise ValueError(f"Unknown family: {family}")
    df_snp = dset.snp.copy()
    df_indiv = dset.indiv.copy()
    columns = []
    for sim_i in range(n_sim):
        columns.extend([f"SIM{sim_i}.ANC{anc_i}" for anc_i in range(dset.n_anc)])

    df_beta = pd.DataFrame(index=df_snp.index, columns=columns)
    # fill in beta
    for anc_i in range(dset.n_anc):
        df_beta.iloc[:, anc_i :: dset.n_anc] = dict_sim["beta"][:, anc_i, :]

    df_pheno = pd.DataFrame(
        dict_sim["pheno"],
        columns=[f"SIM{i}" for i in range(n_sim)],
        index=df_indiv.index,
    )

    for suffix, df in zip(["beta", "pheno"], [df_beta, df_pheno]):
        df.to_csv(f"{out_prefix}.{suffix}", sep="\t", float_format="%.6g")


def simulate_pheno(
    plink_path: str,
    hsq: float,
    out: str,
    var_e: float = None,
    family: str = "quant",
    n_causal: int = None,
    p_causal: float = None,
    case_prevalence: float = 0.5,
    seed: int = None,
    snp_effect: str = None,
    n_sim: int = 10,
):

    """Simulate phenotypes from plink files

    Parameters
    ----------
    plink_path : str
        path to plink file
    hsq : float
        simulated heritability
    var_e: float
        variance of environmental noise
    out : str
        output prefix
    family : str, optional
        distribution family of y, by default "quant"
    n_causal : int, optional
        number of causal variants, by default None
    p_causal : float, optional
        proportion of causal variants, by default None
    case_prevalence : float, optional
        prevalence of cases, by default 0.5
    seed : int, optional
        random seed, by default None
    snp_effect : str, optional
        SNP effects file, by default None
    n_sim : int, optional
        number of simulations, by default 10
    """
    log_params("simulate-pheno", locals())

    ## check input
    assert snp_effect is None, "snp_effect is not supported yet"
    assert var_e is None, "var_e is not supported yet"
    if seed is not None:
        np.random.seed(seed)

    assert (n_causal is not None) + (
        p_causal is not None
    ) == 1, "one of n_causal/p_causal must be specified"

    geno, df_snp, df_indiv = dapgen.read_plink(plink_path)
    if n_causal is None:
        n_causal = int(df_snp.shape[0] * p_causal)

    dict_simu = admix.simulate.quant_pheno_1pop(
        geno=geno, hsq=hsq, n_causal=n_causal, n_sim=n_sim
    )
    columns = [f"SIM{i}" for i in range(n_sim)]
    df_beta = pd.DataFrame(data=dict_simu["beta"], index=df_snp.index, columns=columns)
    df_pheno_g = pd.DataFrame(
        dict_simu["pheno_g"],
        columns=columns,
        index=df_indiv.index,
    )
    df_pheno = pd.DataFrame(
        dict_simu["pheno"],
        columns=columns,
        index=df_indiv.index,
    )
    admix.logger.info(f"results written to {out}.[beta|pheno_g|pheno].tsv")
    for suffix, df in zip(
        ["beta", "pheno_g", "pheno"], [df_beta, df_pheno_g, df_pheno]
    ):
        df.to_csv(f"{out}.{suffix}.tsv", sep="\t", float_format="%.6g")
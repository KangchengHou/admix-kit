#!/usr/bin/env python

import admix
import pandas as pd
import numpy as np
from typing import List
from ._assoc import assoc
from ._utils import log_params


def simulate_pheno(
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
    Simulate phenotypes from a pgen file.

    Parameters
    ----------
    pfile : str
        Path to the pgen file
    hsq : float
        Heritability
    out_prefix : str
        Prefix to the output file, <out>.pheno, <out>.snpeffect will be created
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

    assert not (
        n_causal is not None and p_causal is not None
    ), "`n_causal` and `p_causal` can not be both specified"

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

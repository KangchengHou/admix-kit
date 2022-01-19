import numpy as np
from typing import List
import dask.array as da
import admix
import pandas as pd
from tqdm import tqdm
from ._lanc import hap_lanc as simulate_hap_lanc


def admix_geno(
    geno_list: List[da.Array],
    df_snp: pd.DataFrame,
    anc_props: List[float],
    mosaic_size: float,
    n_indiv: int,
    return_sparse_lanc=False,
) -> admix.Dataset:
    """Simulate admixed genotype

    The generative model is:

    - for each ancestry, the allele frequencies are drawn
    - for each individual, breakpoints are drawn from a Poisson process. and the ancestry
        will be filled based a multinomial distribution with `n_anc` components
    - for each SNP, given the ancestry and the allele frequencies, the haplotype is drawn.
        Haplotype are simulated under some frequencies

    Parameters
    ----------
    geno_list : List[da.Array]
        List of ancestral data sets, each with (n_snp, n_indiv)
    df_snp : pd.DataFrame
        Dataframe of SNPs shared across ancestral data sets
    anc_props : list of float
        Proportion of ancestral populations
    mosaic_size : float
        Expected mosaic size in # of SNPs. use admix.lanc.calculate_mosaic_size() to
        calculate the mosaic size
    n_indiv : int
        Number of individuals to simulate

    Returns
    -------
    admix.Dataset
        Simulated admixed dataset
    """

    n_anc = len(geno_list)
    n_snp = geno_list[0].shape[0]
    assert all(
        n_snp == geno.shape[0] for geno in geno_list
    ), "all geno must have the same number of SNPs"
    assert n_snp == df_snp.shape[0], "df_snp must have the same number of SNPs"

    assert np.sum(anc_props) == 1, "anc_props must sum to 1"
    assert len(anc_props) == n_anc, "anc_props must have the same length as n_anc"
    anc_props = np.array(anc_props)

    dset_hap_list = [
        da.hstack([geno[:, :, 0], geno[:, :, 1]]).compute() for geno in geno_list
    ]

    hap_lanc_breaks, hap_lanc_values = admix.simulate.hap_lanc(
        n_snp=n_snp, n_hap=n_indiv * 2, mosaic_size=mosaic_size, anc_props=anc_props
    )
    geno = np.zeros((n_snp, n_indiv * 2), dtype=np.int8)

    for hap_i in tqdm(range(n_indiv * 2)):
        start = 0
        for stop, val in zip(hap_lanc_breaks[hap_i], hap_lanc_values[hap_i]):
            geno[start:stop, hap_i] = dset_hap_list[val][
                start:stop, np.random.randint(dset_hap_list[val].shape[1])
            ]
            start = stop
    lanc_breaks, lanc_values = admix.data.haplo2diplo(
        breaks=hap_lanc_breaks, values=hap_lanc_values
    )
    lanc = admix.data.Lanc(breaks=lanc_breaks, values=lanc_values)

    # a = np.random.randn(4, 5, 2)
    # b = np.zeros((4, 10))
    # b[:, 0::2] = a[:, :, 0]
    # b[:, 1::2] = a[:, :, 1]
    # assert np.all(b == a.reshape(4, 10))
    # assert np.all(b.reshape(4, 5, 2) == a)

    # equivalent: geno = np.dstack([geno[:, 0::2], geno[:, 1::2]])
    geno = geno.reshape(n_snp, n_indiv, 2)

    dset = admix.Dataset(
        geno=da.from_array(geno, chunks=-1),
        lanc=lanc.dask(snp_chunk=n_snp),
        n_anc=n_anc,
        snp=df_snp,
        indiv=pd.DataFrame(
            {"indiv": ["indiv_" + str(i) for i in np.arange(n_indiv)]}
        ).set_index("indiv"),
    )
    if return_sparse_lanc:
        return dset, lanc
    else:
        return dset


def admix_geno_simple(
    n_indiv: int,
    n_snp: int,
    n_anc: int,
    mosaic_size: float,
    anc_props: np.ndarray,
    allele_freqs: List[np.ndarray] = None,
    af_maf_low: float = 0.05,
    af_maf_high: float = 0.5,
) -> admix.Dataset:
    """Simulate admixed genotype

    The generative model is:

    - for each ancestry, the allele frequencies are drawn
    - for each individual, breakpoints are drawn from a Poisson process. and the ancestry will be filled based a multinomial distribution with `n_anc` components
    - for each SNP, given the ancestry and the allele frequencies, the haplotype is drawn. Haplotype are simulated under some frequencies

    Parameters
    ----------
    n_indiv : int
        Number of individuals
    n_snp : int
        Number of SNPs
    n_anc : int
        Number of ancestries
    mosaic_size : float
        Expected mosaic size in # of SNPs
    anc_props : list of float
        Proportion of ancestral populations, if not specified, the proportion
        is uniform over the ancestral populations.
    allele_freqs: List[np.ndarray]
        Allele frequencies, if not specified, the frequencies are drawn from
        using `af_maf_low` and `af_maf_high`
    af_maf_low : float
        Lowest allowed minor allele frequency
    af_maf_low : float
        Highest allowed minor allele frequency

    Returns
    -------
    admix.Dataset
        Simulated admixed dataset


    References
    ----------
    1. https://github.com/slowkoni/rfmix/blob/master/simulate.cpp
    2. https://github.com/slowkoni/admixture-simulation
    3. https://github.com/williamslab/admix-simu
    """
    if anc_props is None:
        anc_props = np.ones(n_anc) / n_anc
    else:
        anc_props = np.array(anc_props)
        assert np.sum(anc_props) == 1, "anc_props must sum to 1"
    anc_props = np.array(anc_props)
    assert anc_props.size == n_anc, "anc_props must have the same length as n_anc"

    breaks, values = simulate_hap_lanc(
        n_hap=n_indiv * 2, n_snp=n_snp, mosaic_size=mosaic_size, anc_props=anc_props
    )
    breaks, values = admix.data.haplo2diplo(breaks=breaks, values=values)
    rls_lanc = (
        admix.data.Lanc(breaks=breaks, values=values).dask().swapaxes(0, 1).compute()
    )
    # n_indiv x n_snp x 2 (2 for each haplotype)
    # rls_lanc = np.dstack([rls_lanc[0:n_indiv, :], rls_lanc[n_indiv:, :]])
    rls_geno = np.zeros_like(rls_lanc)
    if allele_freqs is None:
        # allele frequencies for the two populations
        allele_freqs = [
            np.random.uniform(low=af_maf_low, high=af_maf_high, size=n_snp)
            for _ in range(n_anc)
        ]
    else:
        assert (
            len(allele_freqs) == n_anc
        ), "allele_freqs must have the same length as n_anc"
        assert np.all(
            [len(af) == n_snp for af in allele_freqs]
        ), "each element in allele_freqs must have the same length as n_snp"

    for i_anc in range(n_anc):
        rls_geno[rls_lanc == i_anc] = np.random.binomial(
            n=1, p=allele_freqs[i_anc][np.where(rls_lanc == i_anc)[1]]
        )

    return admix.Dataset(
        geno=da.from_array(np.swapaxes(rls_geno, 0, 1), chunks=-1),
        lanc=da.from_array(np.swapaxes(rls_lanc, 0, 1), chunks=-1),
        n_anc=n_anc,
    )


def admix_geno2(
    n_indiv: int,
    n_snp: int,
    n_anc: int,
    mosaic_size: float,
    anc_props: np.ndarray = None,
    ancestral_af_range: List[float] = [0.1, 0.9],
    F_st: float = 0.3,
) -> admix.Dataset:
    """Simulate admixed genotype
    The generative model is:
        1. for each ancestry, the allele frequencies are drawn
        2. for each individual, breakpoints are drawn from a Poisson process.
            and the ancestry will be filled based a multinomial distribution with `n_anc`
            components
        3. for each SNP, given the ancestry and the allele frequencies, the haplotype
            is drawn
    Haplotype are simulated under some frequencies

    Parameters
    ----------
    n_indiv : int
        Number of individuals
    n_snp : int
        Number of SNPs
    n_anc : int
        Number of ancestries
    mosaic_size : float
        Expected mosaic size in # of SNPs
    anc_props : list of float
        Proportion of ancestral populations, if not specified, the proportion
        is uniform over the ancestral populations.
    ancestral_af_range: [float, float]
        allele frequencies range.
    F_st: float
        Distance for the two populations that form the admixed population.

    Returns
    -------
        admix.Dataset
        Simulated admixed dataset

    References
    ----------
    We follow Zaitlen et al. 2014, Nature Genetics
    See description in subsection "Simulations with simulated genotypes" in Methods
    """
    assert n_anc == 2, "n_anc must be 2, only two-way admixture is currently supported"
    # global ancestry
    if anc_props is None:
        anc_props = np.ones(n_anc) / n_anc
    else:
        anc_props = np.array(anc_props)
        assert np.sum(anc_props) == 1, "anc_props must sum to 1"
    anc_props = np.array(anc_props)
    assert anc_props.size == n_anc, "anc_props must have the same length as n_anc"

    # local ancestry
    rls_lanc = (
        simulate_hap_lanc(
            n_hap=n_indiv * 2, n_snp=n_snp, mosaic_size=mosaic_size, anc_props=anc_props
        )
        .dask()
        .T
    )
    # n_indiv x n_snp x 2 (2 for each haplotype)
    rls_lanc = np.dstack([rls_lanc[0:n_indiv, :], rls_lanc[n_indiv:, :]])
    rls_geno = np.zeros_like(rls_lanc)

    ancestral_afs = np.random.uniform(
        low=ancestral_af_range[0], high=ancestral_af_range[1], size=n_snp
    )

    # allele frequencies for the two populations
    allele_freqs = [
        np.random.beta(
            a=ancestral_afs * (1 - F_st) / F_st,
            b=(1 - ancestral_afs) * (1 - F_st) / F_st,
        )
        for _ in range(n_anc)
    ]

    for i_anc in range(n_anc):
        rls_geno[rls_lanc == i_anc] = np.random.binomial(
            n=1, p=allele_freqs[i_anc][np.where(rls_lanc == i_anc)[1]]
        )
    return admix.Dataset(
        geno=da.from_array(np.swapaxes(rls_geno, 0, 1)),
        lanc=da.from_array(np.swapaxes(rls_lanc, 0, 1)),
        n_anc=n_anc,
    )

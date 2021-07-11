import numpy as np
import xarray as xr
from typing import List, Collection, Optional
import dask.array as da


def _lanc(
    n_hap: int,
    n_snp: int,
    mosaic_size: float,
    anc_props: List[float],
) -> np.ndarray:
    """Simulate local ancestries based on Poisson process.


    TODO: we could specify the centimorgan for all the SNPs, then infer some
    recombination rates from the centimorgans.
    Parameters
    ----------
    n_hap : int
        Number of haplotypes
    n_snp : int
        Number of SNPs
    mosaic_size : float
        Expected mosaic size in # of SNPs
    anc_props : list of float
        Proportion of ancestral populations, if not specified, the proportion
        is uniform over the ancestral populations.

    Returns
    -------
    np.ndarray
        Simulated local ancestry
    """
    assert np.sum(anc_props) == 1, "anc_props must sum to 1"
    n_total_snp = n_hap * n_snp

    # TODO: infer the parameter in the exponential distribution

    # number of chunks to simulate in each iteration
    chunk_size = int(n_total_snp / mosaic_size)

    breaks: List[float] = []
    while np.sum(breaks) < n_total_snp:
        breaks.extend(np.random.exponential(scale=mosaic_size, size=chunk_size))
    breaks = np.ceil(breaks).astype(int)
    breaks = breaks[0 : np.argmax(np.cumsum(breaks) > n_total_snp) + 1]

    ancestries = np.random.choice(
        np.arange(len(anc_props)), size=len(breaks), p=anc_props
    )

    rls_lanc = np.repeat(ancestries, breaks)[0:n_total_snp].reshape(n_hap, n_snp)

    return rls_lanc


def admix_geno(
    n_indiv: int,
    n_snp: int,
    n_anc: int,
    mosaic_size: float,
    anc_props: np.ndarray = None,
    maf_low: float = 0.05,
    maf_high: float = 0.5,
) -> xr.Dataset:
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
    maf_low : float
        Lowest allowed minor allele frequency
    maf_high : float
        Highest allowed minor allele frequency

    Returns
    -------
        xr.Dataset
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

    rls_lanc = _lanc(n_indiv * 2, n_snp, mosaic_size=mosaic_size, anc_props=anc_props)
    # n_indiv x n_snp x 2 (2 for each haplotype)
    rls_lanc = np.dstack([rls_lanc[0:n_indiv, :], rls_lanc[n_indiv:, :]])
    rls_geno = np.zeros_like(rls_lanc)
    # allele frequencies for the two populations
    allele_freqs = [
        np.random.uniform(low=maf_low, high=maf_high, size=n_snp) for _ in range(n_anc)
    ]
    for i_anc in range(n_anc):
        rls_geno[rls_lanc == i_anc] = np.random.binomial(
            n=1, p=allele_freqs[i_anc][np.where(rls_lanc == i_anc)[1]]
        )

    return xr.Dataset(
        data_vars={
            "geno": (["indiv", "snp", "haploid"], da.from_array(rls_geno)),
            "lanc": (["indiv", "snp", "haploid"], da.from_array(rls_lanc)),
        },
        attrs={
            "n_anc": n_anc,
        },
    )

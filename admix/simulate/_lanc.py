import numpy as np
from typing import List, Tuple
from admix.tools import get_cache_data
import pandas as pd
from bisect import bisect_right, bisect_left


def calculate_mosaic_size(
    df_snp: pd.DataFrame, genetic_map: str, chrom: int, n_gen: int
) -> float:
    """Calculate the expected mosaic size in number of SNPs.

    1. Calculate the length of the region in cM using df_snp and genetic_map
    2. The expected number of cross-over events is calculated as
        n_cross_over = length_in_cM * n_gen
    3. The expected mosaic size is calculated as
        expected_mosaic_size = n_snp / n_cross_over

    Parameters
    ----------
    df_snp: pd.DataFrame
        DataFrame of SNP information
    genetic_map: str
        map to use, either 'hg19' or 'hg38'
    chrom: int
        specify the chromosome df_snp is, df_snp can only have one chromosome
    n_gen: int
        number of generations to simulate
    """

    assert np.all(df_snp.CHROM == chrom)
    assert genetic_map in ["hg19", "hg38"], "genetic_map must be either hg19 or hg38"

    n_snp = len(df_snp)

    # number of centimorgans
    df_map = pd.read_csv(
        get_cache_data("genetic_map", build=genetic_map), delim_whitespace=True
    )
    df_map = df_map[df_map.chr == chrom].drop(columns=["chr"])
    # find the closest SNP to df_snp.POS[0] and df_snp.POS[-1]
    # and calculate the length of the region in cM
    cm_start = df_map["Genetic_Map(cM)"].values[
        np.argmin(np.abs(df_map["position"] - df_snp.POS.values[0]))
    ]
    cm_stop = df_map["Genetic_Map(cM)"].values[
        np.argmin(np.abs(df_map["position"] - df_snp.POS.values[-1]))
    ]
    cm_length = cm_stop - cm_start

    # 1 cross-over per Morgan (centiMorgan / 100)
    expected_mosaic_size = n_snp / (n_gen * cm_length / 100)
    return expected_mosaic_size


def hap_lanc(
    n_snp: int,
    n_hap: int,
    mosaic_size: float,
    anc_props: List[float],
) -> Tuple[List[List[int]], List[List[int]]]:

    """Simulate local ancestries based on Poisson process. The simulated Poisson process
    will be homogeneous because non-homogeneous Poisson process is not easily supported.
    We will take the use df_snp to calculate the length of the region in cM. And assume
    a constant rate of cross-over.

    Haploid data are generated, use admix.data.haplo2diplo to combine pairs of
    haplotypes to get diploid data.

    The simulation process is as follows:
    1. Calculate the length of the region in cM using df_snp and genetic_map
    2. The expected number of cross-over events is calculated as
        n_cross_over = length_in_cM * n_gen
    3. The expected mosaic size is calculated as
        expected_mosaic_size = n_snp / n_cross_over


    Parameters
    ----------
    n_gen: int
        Number of generations to simulate
    n_hap: int
        Number of haplotypes to simulate
    mosaic_size: float
        Expected mosaic size in number of SNPs, use admix.simulate.calculate_mosaic_size
        to compute
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

    # number of chunks to simulate in each iteration, simulate until
    # the desired total number of SNPs is reached
    chunk_size = int(n_total_snp / mosaic_size)

    raw_breaks: List[float] = []
    while np.sum(raw_breaks) < n_total_snp:
        raw_breaks.extend(np.random.exponential(scale=mosaic_size, size=chunk_size))
    breaks = np.cumsum(np.ceil(raw_breaks).astype(int))
    # find first break that is larger than the desired number of SNPs
    breaks = breaks[0 : int(np.argmax(breaks > n_total_snp) + 1)]
    # simulate local ancestry values
    values = np.random.choice(np.arange(len(anc_props)), size=len(breaks), p=anc_props)

    # insert values at n_snp, 2 * n_snp, ...
    boundary_loc = [
        bisect_right(breaks, v) for v in np.arange(1, n_hap + 1).astype(int) * n_snp
    ]

    # ...[:-1] to remove the last chunk because that would correspond to the extra
    # (n_hap + 1) haplotype
    break_list = np.split(breaks, boundary_loc)[:-1]
    value_list = np.split(values, boundary_loc)[:-1]
    # mod(..., n_snp + 1) to retain the boundary case of n_snp
    break_list = [np.mod(br, n_snp + 1).tolist() + [n_snp] for br in break_list]

    # the boundary take the value of next break
    boundary_values = values[boundary_loc]
    value_list = [vl.tolist() + [boundary_values[i]] for i, vl in enumerate(value_list)]

    # shuffle the break_list and value_list
    shuffled = list(zip(break_list, value_list))
    np.random.shuffle(shuffled)
    break_list, value_list = zip(*shuffled)

    return break_list, value_list
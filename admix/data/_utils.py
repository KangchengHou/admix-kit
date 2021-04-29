import numpy as np
import re
from smart_open import open


def read_digit_mat(path, filter_non_numeric=False):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    if filter_non_numeric:
        with open(path) as f:
            mat = np.array(
                [
                    np.array([int(c) for c in re.sub("[^0-9]", "", line.strip())])
                    for line in f.readlines()
                ],
                dtype=np.int8,
            )
    else:
        with open(path) as f:
            mat = np.array(
                [np.array([int(c) for c in line.strip()]) for line in f.readlines()],
                dtype=np.int8,
            )
    return mat


def write_digit_mat(path, mat):
    """
    Read a matrix of integer with [0-9], and with no delimiter.

    Args
    ----

    """
    np.savetxt(path, mat, fmt="%d", delimiter="")


def seperate_ld_blocks(anc, phgeno, legend, ld_blocks):
    assert len(legend) == anc.shape[1]
    assert len(legend) == phgeno.shape[1]

    rls_list = []
    for block_i, block in ld_blocks.iterrows():
        block_index = np.where(
            (block.START <= legend.position) & (legend.position < block.STOP)
        )[0]
        block_legend = legend.loc[block_index]
        block_anc = anc[:, block_index]
        block_phgeno = phgeno[:, block_index]
        rls_list.append((block_anc, block_phgeno, block_legend))
    return rls_list


def convert_anc_count(phgeno: np.ndarray, anc: np.ndarray) -> np.ndarray:
    """
    Convert from ancestry and phased genotype to number of minor alles for each ancestry
    version 2, it should lead to exact the same results as `convert_anc_count`

    Args:
        phgeno (np.ndarray): (n_indiv, 2 x n_snp), the first half columns contain the first haplotype,
            the second half columns contain the second haplotype
        anc (np.ndarray): n_indiv x 2n_snp, match `phgeno`

    Returns:
        np.ndarray: n_indiv x 2n_snp, the first half columns stores the number of minor alleles
        from the first ancestry, the second half columns stores the number of minor
        alleles from the second ancestry
    """
    n_indiv = anc.shape[0]
    n_snp = anc.shape[1] // 2
    n_anc = 2
    geno = np.zeros_like(phgeno)
    for haplo_i in range(2):
        haplo_slice = slice(haplo_i * n_snp, (haplo_i + 1) * n_snp)
        haplo_phgeno = phgeno[:, haplo_slice]
        haplo_anc = anc[:, haplo_slice]
        for anc_i in range(n_anc):
            geno[:, (anc_i * n_snp) : ((anc_i + 1) * n_snp)][
                haplo_anc == anc_i
            ] += haplo_phgeno[haplo_anc == anc_i]

    return geno


def convert_anc_count2(phgeno, anc):
    """
    Convert from ancestry and phased genotype to number of minor alles for each ancestry

    Args
    ----
    phgeno: n_indiv x 2n_snp, the first half columns contain the first haplotype,
        the second half columns contain the second haplotype
    anc: n_indiv x 2n_snp, match `phgeno`

    Returns
    ----
    geno: n_indiv x 2n_snp, the first half columns stores the number of minor alleles
        from the first ancestry, the second half columns stores the number of minor
        alleles from the second ancestry
    """
    n_indiv = anc.shape[0]
    n_snp = anc.shape[1] // 2
    phgeno = phgeno.reshape((n_indiv * 2, n_snp))
    anc = anc.reshape((n_indiv * 2, n_snp))

    geno = np.zeros((n_indiv, n_snp * 2), dtype=np.int8)
    for indiv_i in range(n_indiv):
        for haplo_i in range(2 * indiv_i, 2 * indiv_i + 2):
            for anc_i in range(2):
                anc_snp_index = np.where(anc[haplo_i, :] == anc_i)[0]
                geno[indiv_i, anc_snp_index + anc_i * n_snp] += phgeno[
                    haplo_i, anc_snp_index
                ]
    return geno


def add_up_haplotype(haplo):
    """
    Adding up the values from two haplotypes

    Args
    -----
    haplo: (n_indiv, 2 * n_snp) matrix

    Returns
    -----
    (n_indiv, n_snp) matrix with added up haplotypes
    """
    assert haplo.shape[1] % 2 == 0
    n_snp = haplo.shape[1] // 2
    return haplo[:, np.arange(n_snp)] + haplo[:, np.arange(n_snp) + n_snp]
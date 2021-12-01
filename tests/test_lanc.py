import admix
from admix.data import Lanc
import numpy as np


def test_basic():
    """
    Test basic functionality of Lanc
    """
    # two individuals
    # 1st: 1:01, 2:00, 4:10
    # 2nd: 2:01, 4:00
    # 1st:
    # 0011
    # 1000
    # 2nd
    # 0000
    # 1100
    n_snp, n_indiv = 4, 2
    breaks = [[1, 2, 4], [2, 4]]
    values = [["00", "11", "22"], ["33", "44"]]

    # construct dense matrix
    dense = np.zeros((n_snp, n_indiv, 2))
    dense[:, 0, 0] = [0, 1, 2, 2]
    dense[:, 0, 1] = [0, 1, 2, 2]
    dense[:, 1, 0] = [3, 3, 4, 4]
    dense[:, 1, 1] = [3, 3, 4, 4]

    lanc = Lanc(breaks=breaks, values=values)
    assert np.all(lanc.dask().compute() == dense)
    assert np.all(lanc.numpy() == dense)

    # subset
    assert np.all(lanc[2:4, :].dask().compute() == dense[2:4, :])

    # concat
    assert np.all(
        admix.data.concat_lancs([lanc, lanc[2:4]]).dask()
        == np.vstack([dense, dense[2:4, :]])
    )

    # impute
    # chrom_pos = 1, 4, 7, 10
    chrom_pos = np.zeros((n_snp, 2), dtype=int)
    chrom_pos[:, 0] = 1
    chrom_pos[:, 1] = [1, 4, 7, 10]
    dst_chrom_pos = chrom_pos.copy()

    # dst_chrom_pos = 2, 5, 8, 11
    chrom_pos[:, 1] = [1, 4, 7, 10]
    dst_chrom_pos[:, 1] = [2, 5, 8, 11]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask() == lanc.dask())

    # dst_chrom_pos = 3, 4, 7, 10
    chrom_pos[:, 1] = [1, 4, 7, 10]
    dst_chrom_pos[:, 1] = [3, 4, 7, 10]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)

    # 0th column of dst should map to 1st of src
    assert np.all(dst_lanc.dask()[0, :, :] == lanc.dask()[1, :, :])
    assert np.all(dst_lanc.dask()[1:, :, :] == lanc.dask()[1:, :, :])


def test_lanc_impute():
    """
    More comprehensive test of local ancestry imputation
    """
    n_snp, n_indiv = 3, 1
    breaks = [[1, 2, 3]]
    values = [["01", "23", "45"]]

    # construct dense matrix
    dense = np.zeros((n_snp, n_indiv, 2))
    dense[:, 0, 0] = [0, 1, 0]
    dense[:, 0, 1] = [1, 0, 1]

    lanc = Lanc(breaks=breaks, values=values)

    chrom_pos = np.zeros((n_snp, 2), dtype=int)
    chrom_pos[:, 0] = 1
    chrom_pos[:, 1] = [1, 4, 7]

    dst_chrom_pos = chrom_pos.copy()

    # Scenerio 1
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [3, 4, 7]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0] == lanc.dask()[1])
    assert np.all(dst_lanc.dask()[1:] == lanc.dask()[1:])

    # Scenerio 2
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [2, 5, 8]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask() == lanc.dask())

    # Scenerio 3
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [1, 4, 7]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask() == lanc.dask())

    # Scenerio 4
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [7, 8, 9]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask() == lanc.dask()[-1, :, :])

    # Scenerio 5
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [-2, -1, 0]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask() == lanc.dask()[0, :, :])

    # Scenerio 6
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [0, 1, 7]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0:2, :, :] == lanc.dask()[0, :, :])
    assert np.all(dst_lanc.dask()[2, :, :] == lanc.dask()[2, :, :])

    # Scenerio 7
    chrom_pos[:, 1] = [1, 4, 7]
    dst_chrom_pos[:, 1] = [0, 1, 7]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0:2, :, :] == lanc.dask()[0, :, :])
    assert np.all(dst_lanc.dask()[2, :, :] == lanc.dask()[2, :, :])

    # Scenerio 8
    chrom_pos[:, 1] = [1, 5, 9]
    dst_chrom_pos[:, 1] = [3, 5, 7]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0, :, :] == lanc.dask()[0, :, :])
    assert np.all(dst_lanc.dask()[1, :, :] == lanc.dask()[1, :, :])
    assert np.all(dst_lanc.dask()[2, :, :] == lanc.dask()[1, :, :])

    # Scenerio 9
    # 6 in dst_chrom_pos now is not identified as a break points
    chrom_pos[:, 1] = [1, 8, 9]
    dst_chrom_pos[:, 1] = [1, 6, 9]
    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0, :, :] == lanc.dask()[0, :, :])
    assert np.all(dst_lanc.dask()[1, :, :] == lanc.dask()[1, :, :])
    assert np.all(dst_lanc.dask()[2, :, :] == lanc.dask()[2, :, :])

    # Scenerio 10: two chromosomes
    chrom_pos[:, 0] = [1, 1, 2]
    chrom_pos[:, 1] = [1, 5, 9]
    dst_chrom_pos[:, 0] = [1, 2, 2]
    dst_chrom_pos[:, 1] = [4, 5, 7]

    dst_lanc = lanc.impute(chrom_pos=chrom_pos, dst_chrom_pos=dst_chrom_pos)
    assert np.all(dst_lanc.dask()[0, :, :] == lanc.dask()[1, :, :])
    assert np.all(dst_lanc.dask()[1, :, :] == lanc.dask()[2, :, :])
    assert np.all(dst_lanc.dask()[2, :, :] == lanc.dask()[2, :, :])
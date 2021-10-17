import numpy as np
from numpy import (
    float32,
    tril_indices_from,
)
import pandas as pd
import dask
import dask.array as da


def read_lanc(path: str):

    # TODO: first check input
    # the end must equal to n_snp
    with open(path) as f:
        lines = f.readlines()
    n_snp, n_indiv = [int(i) for i in lines[0].strip().split()]
    lanc_list = [line.strip().split() for line in lines[1:]]
    assert len(lanc_list) == n_indiv

    lanc_mat = da.zeros((n_snp, n_indiv, 2), dtype=np.int8)
    for indiv_i, indiv_lanc in enumerate(lanc_list):
        start = 0
        for l in indiv_lanc:
            a = l.split(":")[0]
            a1, a2 = int(a[0]), int(a[1])
            stop = int(l.split(":")[1])
            lanc_mat[start:stop, indiv_i, 0] = a1
            lanc_mat[start:stop, indiv_i, 1] = a2
            start = stop
    return lanc_mat


def write_lanc(path: str, lanc: da.Array):
    f = open(path, "w")
    f.writelines(f"{lanc.shape[0]} {lanc.shape[1]}\n")
    snp_pos, indiv_pos, ploidy_pos = dask.compute(
        np.where(lanc[1:, :, :] != lanc[0:-1, :, :])
    )[0]

    lines = []
    for indiv_i in range(lanc.shape[1]):
        indiv_snp_pos = (snp_pos[indiv_pos == indiv_i] + 1).tolist() + [lanc.shape[0]]
        indiv_snp_pos = np.sort(np.unique(indiv_snp_pos))
        lines.append(
            " ".join(
                [
                    str(lanc[s - 1, indiv_i, 0].compute())
                    + str(lanc[s - 1, indiv_i, 1].compute())
                    + ":"
                    + str(s)
                    for s in indiv_snp_pos
                ]
            )
        )
    f.writelines("\n".join(lines))
    f.close()
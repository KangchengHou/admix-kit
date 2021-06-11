import numpy as np
from typing import List
from pylampld import LampLD
import xarray as xr

"""
This will be the interface to call various local ancestry inference methods
"""


def lanc(
    sample_dset: xr.Dataset,
    ref_dsets: List[xr.Dataset],
    method="lampld",
    n_proto=6,
    window_size=100,
) -> np.ndarray:
    """Local ancestry inference

    Parameters
    ----------
    sample_dset : xr.Dataset
        [description]
    ref_dsets : List[xr.Dataset]
        [description]
    method : str, optional
        [description], by default "lampld"
    n_proto : int, optional
        [description], by default 6
    window_size : int, optional
        [description], by default 100

    Returns
    -------
    np.ndarray
        Estimated local ancestries
    """

    assert method in ["lampld"]
    n_anc = len(ref_dsets)

    # TODO: perform dataset alignment check
    # assert check_align([sample_dset, *ref_dsets])
    n_snp = sample_dset.dims["snp"]
    n_anc = sample_dset.n_anc
    assert (
        len(ref_dsets) == n_anc
    ), "Length of `ref_dsets` should match `n_anc` in `sample_dset`"

    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=n_proto, window_size=window_size)
    ref_list = [
        np.vstack(
            [
                dset.geno[:, :, 0],
                dset.geno[:, :, 1],
            ]
        )
        for dset in ref_dsets
    ]

    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=6, window_size=300)
    model.set_pos(sample_dset["snp_position"].values)
    model.fit(ref_list)

    est = np.dstack(
        [
            model.infer_lanc(sample_dset["geno"][:, :, 0].compute()),
            model.infer_lanc(sample_dset["geno"][:, :, 1].compute()),
        ]
    )
    return est.astype(np.int8)


def estimate_local_ancestry(
    hap: np.ndarray,
    ref_list: List[np.ndarray],
    pos: np.ndarray,
    n_proto=6,
    window_size=300,
):
    """Local ancestry inference
    Given the haplotype, list of reference panel, infer the local ancestry

    Args:
        hap (np.ndarray): haplotype with local ancestry to be inferred
        ref_list (np.ndarray): list of references
    """

    n_anc = len(ref_list)
    n_snp = hap.shape[1]
    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=n_proto)

    model.fit(ref_list=ref_list)
    inferred = model.predict(hap)

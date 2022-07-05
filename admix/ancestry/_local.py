import numpy as np
from typing import List
import admix

"""
This will be the interface to call various local ancestry inference methods
"""


def lanc_single_chrom(
    sample_dset: admix.Dataset,
    ref_dsets: List[admix.Dataset],
    method="lampld",
    n_proto=6,
    window_size=300,
):
    """Local ancestry inference for data sets of single chromosome

    Parameters
    ----------
    sample_dset : admix.Dataset
        dataset of the sample to be inferred
    ref_dsets : List[admix.Dataset]
        dataset of the reference panel
    method : str, optional
        method used to perform local ancestry inference, by default "lampld"
    n_proto : int, optional
        number of prototypes used in lampld, by default 6
    window_size : int, optional
    """
    from pylampld import LampLD

    assert method in ["lampld"]
    assert len(sample_dset.snp.CHROM.unique()), "Datasets are not of single chromosome"
    assert admix.dataset.is_aligned(
        [sample_dset, *ref_dsets]
    ), "Datasets are not aligned"

    n_anc = len(ref_dsets)
    n_snp = sample_dset.n_snp

    model = LampLD(n_snp=n_snp, n_anc=n_anc, n_proto=n_proto, window_size=window_size)
    ref_list = [
        np.vstack(
            [
                dset.geno[:, :, 0].T,
                dset.geno[:, :, 1].T,
            ]
        )
        for dset in ref_dsets
    ]

    model.set_pos(sample_dset.snp.POS.values)
    model.fit(ref_list)

    est = np.dstack(
        [
            model.infer_lanc(sample_dset.geno[:, :, 0].T.compute()),
            model.infer_lanc(sample_dset.geno[:, :, 1].T.compute()),
        ]
    )
    return np.swapaxes(est.astype(np.int8), 0, 1)


def lanc(
    sample_dset: admix.Dataset,
    ref_dsets: List[admix.Dataset],
    method="lampld",
    n_proto=6,
    window_size=300,
) -> np.ndarray:
    """Local ancestry inference

    Parameters
    ----------
    sample_dset : admix.Dataset
        dataset of the sample to be inferred
    ref_dsets : List[admix.Dataset]
        dataset of the reference panel
    method : str, optional
        method used to perform local ancestry inference, by default "lampld"
    n_proto : int, optional
        number of prototypes used in lampld, by default 6
    window_size : int, optional
        window size used in local ancestry inference, by default 100

    Returns
    -------
    np.ndarray
        Estimated local ancestries
    """

    assert method in ["lampld"]
    chrom_list = sample_dset.snp.CHROM.unique()
    lanc = []
    for chrom in chrom_list:
        chrom_sample_dset = sample_dset[(sample_dset.snp.CHROM == chrom).values]
        chrom_ref_dsets = [dset[(dset.snp.CHROM == chrom).values] for dset in ref_dsets]
        chrom_lanc = lanc_single_chrom(
            chrom_sample_dset,
            chrom_ref_dsets,
            method=method,
            n_proto=n_proto,
            window_size=window_size,
        )
        lanc.append(chrom_lanc)
    return np.concatenate(lanc, axis=0)

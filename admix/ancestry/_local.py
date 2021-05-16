import numpy as np
from typing import List
from pylampld import LampLD

"""
This will be the interface to call various local ancestry inference methods
"""


def estimate_local_ancestry(
    hap: np.ndarray, ref_list: List[np.ndarray], pos:np.ndarray, n_proto=6, window_size=300
):
    """Local ancestry inference
    Given the haplotype, list of reference panel, infer the local ancestry

    Args:
        hap (np.ndarray): haplotype with local ancestry to be inferred
        ref_list (np.ndarray): list of references
    """

    n_anc = len(ref_list)
    n_snp = hap.shape[1]
    lampld = LampLD(
        n_snp=n_snp,
        n_anc=n_anc,
        n_proto=n_proto
    )

    lampld.fit(ref_list=ref_list)
    inferred = lamp.predict(hap)

import numpy as np
import pandas as pd
from typing import List
from ._lamp import Lamp

"""
This will be the interface to call various local ancestry inference methods
"""


def infer_local_ancestry(
    hap: np.ndarray, ref_list: List[np.ndarray], n_proto=6, window_size=300
):
    """Local ancestry inference
    Given the haplotype, list of reference panel, infer the local ancestry

    Args:
        hap (np.ndarray): haplotype with local ancestry to be inferred
        ref_list (np.ndarray): list of references
    """

    n_anc = len(ref_list)
    n_snp = hap.shape[1]
    lamp = Lamp(
        n_anc=n_anc,
        n_snp=n_snp,
        window_size=window_size,
        n_proto=n_proto,
    )
    lamp.fit(ref_list=ref_list)
    inferred = lamp.predict(hap)

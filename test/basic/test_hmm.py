import numpy as np
import pandas as pd
import admix

import os


def test_basic():
    # test convert anc count
    lanc = np.array([[0, 0, 0, 0, 0, 0]])
    hap = np.array([[0, 1, 0, 1, 1, 0]])
    geno1 = admix.data.convert_anc_count(phgeno=hap, anc=lanc)
    geno2 = admix.data.convert_anc_count2(phgeno=hap, anc=lanc)
    assert np.all(geno1 == [1, 2, 0, 0, 0, 0])
    assert np.all(geno2 == [1, 2, 0, 0, 0, 0])
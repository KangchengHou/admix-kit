import numpy as np
import pandas as pd
from admix.data import read_lanc, read_haplo, convert_anc_count, convert_anc_count2

import os

# The test data is constructed as follows
"""
import numpy as np
import pandas as pd
anc = np.load("anc.npy")
phgeno = np.load("phgeno.npy")
legend = pd.read_csv("legend.csv")

n_snp = legend.shape[0]
n_haplo = anc.shape[0]

snp_index = np.linspace(0, n_snp - 1, 200).astype(int)
haplo_index = np.arange(1000)
legend.iloc[snp_index, :].to_csv("test.legend", index=False)
np.savetxt("test.lanc", anc[np.ix_(haplo_index, snp_index)], fmt="%d", delimiter='')
np.savetxt("test.haplo", phgeno[np.ix_(haplo_index, snp_index)], fmt="%d", delimiter='')
"""

def test_basic():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    lanc = read_lanc(os.path.join(THIS_DIR, "data/test.lanc"))
    haplo = read_haplo(os.path.join(THIS_DIR, "data/test.haplo"))
    legend = pd.read_csv(os.path.join(THIS_DIR, "data/test.legend"))
    # basic shape
    assert np.all(lanc.shape == haplo.shape)
    assert legend.shape[0] == lanc.shape[1]
    
def test_utils():
    # test convert anc count
    lanc = np.array([[0, 0, 0, 0, 0, 0]])
    haplo = np.array([[0, 1, 0, 1, 1, 0]])
    geno1 = convert_anc_count(phgeno=haplo, anc=lanc)
    geno2 = convert_anc_count2(phgeno=haplo, anc=lanc)
    assert np.all(geno1 == [1, 2, 0, 0, 0, 0])
    assert np.all(geno2 == [1, 2, 0, 0, 0, 0])
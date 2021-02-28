import numpy as np
import pandas as pd
import admix

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
    lanc = admix.data.read_lanc(os.path.join(THIS_DIR, "data/test.lanc"))
    hap = admix.data.read_hap(os.path.join(THIS_DIR, "data/test.hap"))
    legend = pd.read_csv(os.path.join(THIS_DIR, "data/test.legend"))
    # basic shape
    assert np.all(lanc.shape == hap.shape)
    assert legend.shape[0] == lanc.shape[1]

    n_hap, n_snp = hap.shape
    hap = hap.reshape(n_hap // 2, n_snp * 2)
    lanc = lanc.reshape(n_hap // 2, n_snp * 2)
    # phenotype simulation
    sim_pheno = admix.simulate.simulate_phenotype_case_control_1snp(
        hap=hap, lanc=lanc, case_prevalence=0.1, odds_ratio=1.0, n_sim=10
    )
    assert len(sim_pheno) == n_snp
    assert np.all(sim_pheno[0].shape == (n_hap // 2, 10))


def test_utils():
    # test convert anc count
    lanc = np.array([[0, 0, 0, 0, 0, 0]])
    hap = np.array([[0, 1, 0, 1, 1, 0]])
    geno1 = admix.data.convert_anc_count(phgeno=hap, anc=lanc)
    geno2 = admix.data.convert_anc_count2(phgeno=hap, anc=lanc)
    assert np.all(geno1 == [1, 2, 0, 0, 0, 0])
    assert np.all(geno2 == [1, 2, 0, 0, 0, 0])

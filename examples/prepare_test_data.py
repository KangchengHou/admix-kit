"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""

import numpy as np
from admix.data import read_digit_mat, write_digit_mat
import pandas as pd
from os.path import join
src_dir = "/u/project/pasaniuc/pasaniucdata/admixture/kangcheng/genotype_simulation/out/kg_3k"
out_dir = "out"
afr_hap = np.loadtxt(join(src_dir, "AFR.hap"), dtype=np.int8)
eur_hap = np.loadtxt(join(src_dir, "EUR.hap"), dtype=np.int8)
legend = pd.read_csv(join(src_dir, "legend.txt"), delim_whitespace=True)

admix_dir = join(src_dir, "EUR_0.2_AFR_0.8_7_20000")

admix_lanc = read_digit_mat(join(admix_dir, "admix.hanc"))
admix_hap = read_digit_mat(join(admix_dir, "admix.phgeno")).T

write_digit_mat(join(out_dir, "afr.hap"), afr_hap)
write_digit_mat(join(out_dir, "eur.hap"), eur_hap)
write_digit_mat(join(out_dir, "admix.hap"), admix_hap)
write_digit_mat(join(out_dir, "admix.lanc"), admix_lanc)
legend.to_csv(join(out_dir, "legend.csv"), index=False)


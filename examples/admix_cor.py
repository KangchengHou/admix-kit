#%%
# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from admix.simulate import simulate_hap, simulate_lanc, simulate_phenotype_continuous
from admix.plot import plot_local_anc
from matplotlib import collections as mc


# %%
# simulate local ancestry and haplotype

n_indiv = 1000
n_snp = 1000

lanc = simulate_lanc(n_indiv * 2, n_snp, 100)
hap = simulate_hap(n_indiv * 2, n_snp)

ax = plot_local_anc(lanc[0:10, :])
plt.show()


# %%
# simulate haplotype
sim = simulate_phenotype_continuous(
    hap=hap.reshape((n_indiv, n_snp * 2)),
    lanc=lanc.reshape((n_indiv, n_snp * 2)),
    h2g=0.1,
    n_causal=1,
)

# %%
# perform inference

"""
Simulation of genotype, local ancestry and phenotype
===========================

We start by simulating genotype, local ancestry and phenotype
"""

import admix

# from admix.simulate import simulate_hap, simulate_lanc
import matplotlib.pyplot as plt

# %%
# We simulate `n_indiv` individuals and `n_snp` SNPs.

n_indiv = 1000
n_snp = 1000

# %%
# We use function :meth:`~admix.simulate.simulate_lanc`, :meth:`~admix.simulate.simulate_hap` to simulate a few individuals
# lanc = simulate_lanc(n_indiv, n_snp, 100)
# hap = simulate_hap(n_indiv, n_snp)

# print(lanc.shape)
# print(hap.shape)


# ax = admix.plot.lanc(lanc[0:10, :])
# plt.show()


# sim = simulate_continuous_phenotype(
#     hap=hap.reshape((n_indiv, n_snp * 2)),
#     lanc=lanc.reshape((n_indiv, n_snp * 2)),
#     h2g=0.1,
#     n_causal=1,
# )
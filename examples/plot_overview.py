"""
An overview of admix-tools
=======================================

We start with a toy dataset, which contains genotypes of admixed individuals with 
european and african ancestries. We will go through the following:

1. Infer local ancestry.
2. Simulate phenotypes.
3. Perform GWAS.
4. Predict phenotype.
"""

# %%
import admix
import numpy as np
import matplotlib.pyplot as plt


# %%
# We load individuals from both admixed and ancestral population

admix_dset, eur_dset, afr_dset = admix.data.load_toy()

# %%
print(admix_dset)

# %%
print(eur_dset)

# %%
print(afr_dset)


# %%
# These are :class:`xarray.Dataset` object, which is multi-dimensional annotated
# dataset. In our application, they will have 3 dimensions.
#
# * ``indiv``: the dimension of different individual
# * ``snp``: the dimension of SNP
# * ``ploidy``: the dimension of ploidy, which will be always 2.
#
# These :class:`xarray.Dataset` objects will be the central data structure we
# operate on.

# Next, let's infer local ancestries.
est_lanc = admix.ancestry.lanc(
    sample_dset=admix_dset,
    ref_dsets=[eur_dset, afr_dset],
    method="lampld",
    n_proto=8,
    window_size=50,
)
admix_dset = admix_dset.assign(est_lanc=(("indiv", "snp", "ploidy"), est_lanc))
acc = (admix_dset.est_lanc == admix_dset.lanc).mean().compute().item()
print(f"Accuracy: {acc:.2f}")


# %%
# We get a pretty high accuracy using a small reference panel.
# In general, we would get quite accurate estimate of local ancestries.
# In the following, we would just use the groundtruth ``admix_dset.lanc`` stored in the toy
# dataset. In the real data analysis, one could ues the above approach to get ``est_lanc``,
# and assign to ``admix_dset`` with ``admix_dset.lanc = est_lanc``.
#
# We could also plot the groundtruth v.s. estimated local ancestries.

fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
# TODO: make lanc accomodate to dset format directly.
admix.plot.lanc(admix_dset, ax=ax[0])
admix.plot.lanc(admix_dset, ax=ax[1])
ax[0].set_title("Simulated ground-truth")
ax[1].set_title("Estimated")
plt.tight_layout()
plt.show()

# %%
# Now we simulate phenotype for these admixed individuals.
# Maybe simulate case control would be a good idea
np.random.seed(1234)
sim = admix.simulate.continuous_pheno(
    dset=admix_dset, var_g=0.8, var_e=0.2, n_causal=2, n_sim=10
)
beta, pheno_g, pheno = sim["beta"], sim["pheno_g"], sim["pheno"]
print(beta.shape)  # (n_snp, n_anc, n_sim)
print(pheno_g.shape)  # (n_indiv, n_sim)
print(pheno.shape)  # (n_indiv, n_sim)

# TODO: explain what these terms in math, and explain why there is ``n_anc`` in beta

# %%
# Now we perform GWAS with the first set of simulation
i_sim = 0
sim_beta = beta[:, :, i_sim]
sim_pheno = pheno[:, i_sim]

assoc = admix.assoc.marginal(
    dset=admix_dset.assign_coords(pheno=("indiv", sim_pheno)),
    pheno="pheno",
    method="ATT",
    family="linear",
)

# %%
fig, ax = plt.subplots(nrows=2, figsize=(5, 3), dpi=150, sharex=True)
n_snp = admix_dset.dims["snp"]
ax[0].scatter(np.arange(n_snp), -np.log10(assoc.P), s=2)
ax[0].set_ylabel(r"$-\log_{10}(p)$")
ax[0].axhline(
    y=-np.log10(0.05 / n_snp), color="red", ls="--", alpha=0.5
)  # genome-wide significance threshold
ax[1].scatter(np.arange(n_snp), sim_beta[:, 0], s=2)
ax[1].set_ylabel(r"$\beta$")
plt.xlabel("SNP index")
plt.tight_layout()
plt.show()
# %%
# We notice that GWAS correctly points out the first causal SNP

# %%
# TODO: Finally, we perform simple PRS

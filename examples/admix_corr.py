"""
Estimating genetic correlation within admixed individuals
=========================================================

"""

# %%
import admix
import dask

# %%
# Math of the genetic correlation.
# $$
# y = X \alpha + G_1 \beta_1 + G_2 \beta_2 + \epsilon
# $$
# And we assume $\beta$ for each SNP follow a 2D normal distribution
# $$
# \begin{bmatrix}
# \beta_{1j} \\
# \beta_{2j}
# \end{bmatrix}
# \sim \mathcal{N}\left(
# \begin{bmatrix}
# 0\\
# 0
# \end{bmatrix},\begin{bmatrix}
# \sigma_{g1}^{2}/M & \gamma/M\\
# \gamma/M & \sigma_{g2}^{2}/M
# \end{bmatrix}\right),j=1,\dots,M
# $$

# The parameters $\sigma_{g1}^{2}, \sigma_{g1}^{2}, \gamma$ are of interest.

# In the follows, we will first simulate some data where we know the groundtruth, and we
# will apply our methods and show that the method recover these parameters.
# %%
dset = admix.simulate.admix_geno(
    n_indiv=5000, n_snp=1000, n_anc=2, mosaic_size=100, anc_props=[0.1, 0.9]
)
admix.tools.allele_per_anc(dset)
admix.tools.admix_grm(dset, center=True)

grms = {
    "A1": dset["admix_grm_K1"].data + dset["admix_grm_K2"].data,
    "A2": dset["admix_grm_K12"].data + dset["admix_grm_K12"].data.T,
}
grms = dask.persist(grms)[0]

sim = admix.simulate.quant_pheno_grm(
    dset, grm=grms, var={"A1": 1.0, "A2": 0.5, "e": 1.0}, n_sim=50
)
df_rls = admix.estimate.gen_cor(dset, grm=grms, pheno=sim["pheno"], cov_intercept=True)

# %%
# Estimation results for top few rows of simulation
df_rls.head()

#%%
# Expected values across simulations
df_rls.mean(axis=0)

# %%
# Standard deviation across simulations
df_rls.std(axis=0)

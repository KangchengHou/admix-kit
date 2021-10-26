from ._pheno import continuous_pheno, continuous_pheno_1pop, continuous_pheno_grm

# from ._pheno import case_control_pheno


from ._geno import admix_geno

# from ._genotype import simulate_hap

__all__ = [
    "continuous_pheno",
    "case_control_pheno",
    "continuous_pheno_1pop",
    "continuous_pheno_grm",
    "admix_geno",
]

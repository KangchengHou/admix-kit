from ._pheno import quant_pheno, quant_pheno_1pop, quant_pheno_grm

# from ._pheno import case_control_pheno


from ._geno import admix_geno

# from ._genotype import simulate_hap

__all__ = [
    "quant_pheno",
    "binary_pheno",
    "quant_pheno_1pop",
    "quant_pheno_grm",
    "admix_geno",
]

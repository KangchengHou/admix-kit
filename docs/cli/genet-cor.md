# Genetic correlation estimation

### Step 0: Prepare data
1. Phased genotypes and inferred local ancestry (please follow [preparing dataset](../prepare-dataset.md)). So you have `${prefix}.chr${chrom}.[pgen|psam|pvar|lanc]` files.
2. Phenotype and covariates file per trait `${trait}.txt`.

With these files, you can run the following command to estimate $r_\text{admix}$.
### Step 1: compute GRM $\mathbf{K}_1$ and $\mathbf{K}_2$ for each chromosome

```bash
mkdir -p ${out_dir}/admix-grm
admix admix-grm \
    --pfile ${prefix}.chr${chrom} \
    --out-prefix ${out_dir}/admix-grm/chr${chrom}
```
This step will generate `${out_dir}/admix-grm/chr${chrom}.[grm.bin|grm.id|grm.n|weight.tsv]` files.

### Step 2: merging GRMs across chromosomes

```bash
admix admix-grm-merge \
    --prefix ${out_dir}/admix-grm/chr\
    --out-prefix ${out_dir}/admix-grm/merged
```
This step will generate `${out_dir}/admix-grm/merged.[grm.bin|grm.id|grm.n|weight.tsv]` files.

### Step 3: calculating the GRM ($\mathbf{K}_1 + r_\text{admix} \mathbf{K}_2)$ at different $r_\text{admix}$ values and estimating log-likelihood at different $r_\text{admix}$ values

```bash
admix genet-cor \
    --pheno ${trait}.txt
    --grm-prefix ${out_dir}/admix-grm/merged \
    --out-dir ${out_dir}/estimate/${trait}
```


## Parameter options
```{eval-rst}
.. autofunction:: admix.cli.admix_grm
.. autofunction:: admix.cli.admix_grm_merge
.. autofunction:: admix.cli.genet_cor
```


## Additional notes

For more background, we recommend reading [Causal effects on complex traits are similar across segments of different continental ancestries within admixed individuals](https://www.medrxiv.org/content/10.1101/2022.08.16.22278868v1). Nature Genetics (2023).

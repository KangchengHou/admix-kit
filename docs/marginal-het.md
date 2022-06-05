# Evaluating heterogeneity at marginal effects

## Prepare the input data

```python
df_trait = pd.read_csv(<pheno_path>, sep='\t', index_col=0)
# index name should be #IID so that plink2 could recognize
df_trait.index.name = "#IID"
df_trait.iloc[:, [0]].to_csv(f"{out_dir}/pheno.tsv", sep='\t', na_rep='NA')
df_trait.iloc[:, 1:].to_csv(f"{out_dir}/covar.tsv", sep='\t', na_rep='NA')
```


## Step 1: association testing
```bash
plink2 \
    --bfile <bfile> \
    --pheno iid-only <pheno> \
    --covar iid-only <covar> \
    --quantile-normalize \
    --glm hide-covar omit-ref --vif 100 \
    --memory 20000 \
    --out <out>

# the suffix ".PHENO.glm.linear" can change depending on your phenotype
# adjust accordingly (based on PLINK output)
# here we replace ID with SNP so that PLINK1 can recognize the SNP column
sed '1 s/ID/SNP/' <out>.PHENO.glm.linear > <out>.assoc
```


## Step 2: LD clumping
```bash
# <out>.clumped will be created
plink \
    --bfile <bfile> \
    --clump <out>.assoc \
    --clump-p1 5e-8 \
    --clump-p2 1e-4 \
    --clump-r2 0.1 \
    --clump-kb 10000 \
    --memory 20000 \
    --out <out>

# get list of clumped SNPs
awk '(NF > 0) && (NR > 1) {print $3 }' <out>.clumped > <out>.clumped.snp_list
```

## Step 3: evaluating heterogeneity
```bash
# <out>.HET.assoc swill be created
admix assoc \
    --pfile <pfile> \
    --pheno <pheno> \
    --covar <covar> \
    --out <out> \
    --method HET \
    --family quant \
    --quantile-normalize True \
    --snp-list <out>.clumped.snp_list
```
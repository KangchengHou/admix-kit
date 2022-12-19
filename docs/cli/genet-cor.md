# Genetic correlation estimation
For more background on this module, we recommend reading [Causal effects on complex traits are similar across segments of different continental ancestries within admixed individuals](https://www.medrxiv.org/content/10.1101/2022.08.16.22278868v1). medRxiv (2022).
## Technical details
### Problem description
Here, we describe how to estimate $r_\text{admix}$, the genetic correlation of causal effects across local ancestry backgrounds. 

The phenotype of a given admixed individual $y$ is modeled as function of allelic effect sizes that are allowed to vary across ancestries as:

$$
y = \mathbf{g}_1^\top \boldsymbol{\beta}_1 + \mathbf{g}_2^\top \boldsymbol{\beta}_2 + \epsilon, \qquad (1)
$$

where $\mathbf{g}_1, \mathbf{g}_2 \in \mathbb{R}^S$ are the ancestry-specific genotypes for local ancestry 1 and 2, and $\boldsymbol{\beta}_1, \boldsymbol{\beta}_2 \in \mathbb{R}^S$ are the ancestry-specific causal effects ($S$ is the number of genome-wide causal SNPs).

We assume the ancestry-specific causal effects follow the distribution of

$$
\begin{bmatrix} 
\beta_{1s} \\
\beta_{2s}
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
0\\
0
\end{bmatrix},\begin{bmatrix}
\sigma_{g}^{2}/M & \rho_g/S\\
\rho_g/S & \sigma_{g}^{2}/S
\end{bmatrix}\right),s=1,\dots,S, \qquad (2)
$$

and the genetic correlation is defined as $r_\text{admix} = \frac{\rho_g}{\sigma_{g}^{2}}$.

### Estimating $r_\text{admix}$

Extending Equation (1) to $N$ individuals with phenotypes $\mathbf{y} \in \mathbb{R}^N$ and ancestry-specific genotypes $\mathbf{G}_1, \mathbf{G}_2 \in \mathbb{R}^{N \times S}$, we have

$$
\mathbf{y} = \mathbf{G}_1 \boldsymbol{\beta}_1 + \mathbf{G}_2 \boldsymbol{\beta}_2 + \boldsymbol{\epsilon}, \qquad (3)
$$

Integrating out $\boldsymbol{\beta}_1, \boldsymbol{\beta}_2$ and $\boldsymbol{\epsilon}$, we have

$$
\mathbf{y} \sim \mathcal{N}\left(
\mathbf{0}, \sigma_g^2 \mathbf{K}_1 + \rho_g \mathbf{K}_2 + \sigma_e^2 \mathbf{I} \right), \qquad (4)
$$

where we have defined the local ancestry-aware GRM matrices are $\mathbf{K}_1 = \frac{\mathbf{G}_1 \mathbf{G}_1^\top + \mathbf{G}_2 \mathbf{G}_2^\top }{S}, \mathbf{K}_2 = \frac{\mathbf{G}_1 \mathbf{G}_2^\top + \mathbf{G}_2 \mathbf{G}_1^\top }{S}$. We also note that Equation (4) can be rewritten as

$$
\mathbf{y} \sim \mathcal{N}\left(
\mathbf{0}, \sigma_g^2 (\mathbf{K}_1 + r_\text{admix} \mathbf{K}_2) + \sigma_e^2 \mathbf{I} \right). \qquad (4)
$$

Therefore, we can calculate $\mathbf{K}(r_\text{admix}) = \mathbf{K}_1 + r_\text{admix} \mathbf{K}_2$ for a grid of values of $r_\text{admix}$, and use the likelihood ratio curve to estimate $r_\text{admix}$.

## Step 0: Prepare data
1. Phased genotypes and inferred local ancestry (please follow [preparing dataset](../prepare-dataset.md)). So you have `${prefix}.chr${chrom}.[pgen|psam|pvar|lanc]` files.
2. Phenotype and covariates file per trait `${trait}.txt`.

With these files, you can run the following command to estimate $r_\text{admix}$.
## Step 1: compute GRM $\mathbf{K}_1$ and $\mathbf{K}_2$ for each chromosome

```bash
mkdir -p ${out_dir}/admix-grm
admix admix-grm \
    --pfile ${prefix}.chr${chrom} \
    --out-prefix ${out_dir}/admix-grm/chr${chrom}
```
This step will generate `${out_dir}/admix-grm/chr${chrom}.[grm.bin|grm.id|grm.n|weight.tsv]` files.

## Step 2: merging GRMs across chromosomes

```bash
admix admix-grm-merge \
    --prefix ${out_dir}/admix-grm/chr\
    --out-prefix ${out_dir}/admix-grm/merged
```
This step will generate `${out_dir}/admix-grm/merged.[grm.bin|grm.id|grm.n|weight.tsv]` files.

## Step 3: calculating the GRM ($\mathbf{K}_1 + r_\text{admix} \mathbf{K}_2)$ at different $r_\text{admix}$ values and estimating log-likelihood at different $r_\text{admix}$ values

```bash
admix genet-cor \
    --pheno ${trait}.txt
    --grm-prefix ${out_dir}/admix-grm/merged \
    --out-dir ${out_dir}/estimate/${trait}
```

## Reference

```{eval-rst}
.. autofunction:: admix.cli.admix_grm
.. autofunction:: admix.cli.admix_grm_merge
.. autofunction:: admix.cli.genet_cor
```


# Examples
We use an example to go through the pipeline.

## Step 1: Download data
```bash
wget "https://www.dropbox.com/s/ub9c6l82ek2yq8q/admix-simu-data.zip?dl=1" -O admix-simu-data.zip
unzip admix-simu-data.zip
pfile=admix-simu-data/CEU-YRI # for an example of 3-way admixture, use pfile=admix-simu-data/CEU-YRI-PEL
out_dir=out/
mkdir -p ${out_dir}
```

## Step 2: Simulate phenotype
```bash
for cor in 0.9 0.95 1.0; do
    admix simulate-admix-pheno \
        --pfile ${pfile} \
        --hsq 0.25 \
        --p-causal 1.0 \
        --cor ${cor} \
        --n-sim 10 \
        --seed 1234 \
        --out-prefix ${out_dir}/cor-${cor}
done
```

```python
import pandas as pd
import numpy as np

for cor in [0.9, 0.95, 1.0]:
    df = pd.read_csv(f"{out_dir}/cor-{cor}.pheno", sep="\t", index_col=0)
    for i in range(10):
        df_sim = df[[f"SIM{i}"]].copy()
        # add random covariates
        df_sim["COVAR"] = np.random.normal(size=df_sim.shape[0])
        df_sim.to_csv(f"{out_dir}/cor-{cor}.sim{i}.pheno", sep="\t", header=True)
```

## Step 3: Compute GRM
```bash
mkdir -p ${out_dir}/admix-grm
admix append-snp-info \
    --pfile ${pfile} \
    --out ${pfile}.snp_info

admix admix-grm \
    --pfile ${pfile} \
    --out-prefix ${out_dir}/admix-grm/grm
```

## Step 4: Estimate genetic correlation
```bash
cor=0.9
i=0
mkdir -p ${out_dir}/estimate
# this step will take a while
admix genet-cor \
    --pheno ${out_dir}/cor-${cor}.sim${i}.pheno \
    --grm-prefix ${out_dir}/admix-grm/grm \
    --out-dir ${out_dir}/estimate/cor-${cor}.sim${i}

admix summarize-genet-cor \
    --est-dir ${out_dir}/estimate/cor-${cor}.sim${i} \
    --out-prefix ${out_dir}/estimate/cor-${cor}.sim${i}

cat out/estimate/cor-0.9.sim0.summary.json
```

```json
{
    "n": 5000,
    "rg_mode": 0.909,
    "rg_hpdi(50%)": [
        0.854,
        0.952
    ],
    "rg_hpdi(95%)": [
        0.729,
        1.0
    ],
    "rg=1_pval": 0.13
}
```
```{note}
Here the wide credible interval is due to the small sample size (N=5,000) used in the analysis.
```

To obtain results for simulations from all correlation parameters and simulation replicates (or for all traits in real data analysis), we recommend using computing clusters to parallelize this process. After these results are obtained, one can use `admix meta-analyze-genet-cor` to meta-analyze these results. For example,

```bash
for cor in 0.9 0.95 1.0; do
    admix meta-analyze-genet-cor --loglkl-files "out/estimate/cor-${cor}.sim*.loglkl.txt"
done
```

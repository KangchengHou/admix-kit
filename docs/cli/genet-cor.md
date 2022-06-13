# Genetic correlation estimation

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

## Step 2: merging GRMs across chomosomes

```bash
admix admix-grm-merge \
    --prefix ${out_dir}/admix-grm/chr\
    --out-prefix ${out_dir}/admix-grm/merged
```
This step will generate `${out_dir}/admix-grm/merged.[grm.bin|grm.id|grm.n|weight.tsv]` files.

## Step 3: calculating the GRM ($\mathbf{K}_1 + r_\text{admix} \mathbf{K}_2)$ at different $r_\text{admix}$ values

```bash
admix admix-grm-rho \
    --prefix ${out_dir}/admix-grm/merged \
    --out-folder ${out_dir}/admix-grm/rgrid
```

## Step 4: estimating log-likelihood at different $r_\text{admix}$ values

```bash
mkdir -p ${out_dir}/estimate/${trait}
admix estimate-genetic-cor \
    --pheno ${trait}.txt \
    --out-dir ${out_dir}/estimate/${trait} \
    --grm-dir ${out_dir}/admix-grm/rgrid
```

## Reference

```{eval-rst}
.. autofunction:: admix.cli.admix_grm
.. autofunction:: admix.cli.admix_grm_merge
.. autofunction:: admix.cli.admix_grm_rho
.. autofunction:: admix.cli.estimate_genetic_cor
```

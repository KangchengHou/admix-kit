# Genetic correlation estimation

TODO: some introduction of the method, have some math showcasing what each step is calculating.

TODO: specify the overall structure

$$
y = \mathbf{g}_1^\top \boldsymbol{\beta}_1 + \mathbf{g}_2^\top \boldsymbol{\beta}_2 + \epsilon
$$
And we assume $\beta$ for each SNP follow a 2D normal distribution

$$
\begin{bmatrix} 
\beta_{1j} \\
\beta_{2j}
\end{bmatrix}
\sim \mathcal{N}\left(
\begin{bmatrix}
0\\
0
\end{bmatrix},\begin{bmatrix}
\sigma_{g}^{2}/M & \rho/M\\
\gamma/M & \sigma_{g2}^{2}/M
\end{bmatrix}\right),j=1,\dots,M
$$

The parameters $\sigma_{g}^{2}, \sigma_e^2, \rho$ are of interest.

In the follows, we will first simulate some data where we know the groundtruth, and we 
will apply our methods and show that the method recover these parameters.
## Step 1: compute GRM

Example:
```bash
admix admix-grm \
    --pfile <pfile> \
    --out-prefix <out-prefix>
```

## Step 2: merging GRMs across chomosomes

Example:
```bash
admix admix-grm-merge \
    --grm-files <grm-files> \
    --out-prefix <out-prefix>
```

## Step 3: calculating the GRM at different $\rho$ values

Example:
```bash
admix admix-grm-rho \
    --grm-file <grm-file> \
    --out-prefix <out-prefix>
```

## Reference

```{eval-rst}
.. autofunction:: admix.cli.admix_grm
.. autofunction:: admix.cli.admix_grm_rho
.. autofunction:: admix.cli.admix_grm_merge
.. autofunction:: admix.cli.estimate_genetic_cor
```

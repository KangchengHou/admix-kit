# Genetic correlation estimation

TODO: some introduction of the method, have some math showcasing what each step is calculating.

TODO: specify the overall structure
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

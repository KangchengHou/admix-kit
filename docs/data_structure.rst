We use `xr.Dataset` as a container for data. We pose some addtional requirements to `xr.Dataset`:
- It must have `snp` and `indiv` coordinates.
- Each variable on the `snp` coordinate will have `@snp` as the suffix.
- Each variable on the `indiv` coordinate will have `@indiv` as the suffix.
- `n_anc` should appear in the attrs to indicate the number of ancestral populations.

We have a function `admix.data.check_dataset` to check whether a dataset satisfies these 
requirements.
# Association testing

```bash
admix assoc \
    --pfile <geno_file_prefix> \
    --pheno <pheno_file> \
    --method ATT,TRACTOR \
    --family quant \
    --quantile-normalize True \
    --out toy-admix
```

## Parameter options

```{eval-rst}
.. autofunction:: admix.cli.assoc
```


## Additional notes

### Running parallel jobs
To parallelize the analysis, use `--snp-list` option to split the analysis into
multiple jobs with each job analyzing a subset of SNPs. `--snp-list` accepts a 
file path containing a list of SNPs (1 SNP per line). For example, to split the
above job into 10 jobs, we run the following code to create snplist files:

```python
import admix
import numpy as np

DSET_PREFIX="/path/to/pfile" # e.g. "toy-admix"

dset = admix.io.read_dataset(DSET_PREFIX)
index_list = np.array_split(dset.snp.index, 10)
for i, index in enumerate(index_list):
    np.savetxt(f"cache/{DSET_PREFIX}.{i}.snplist".format(i), index, fmt="%s")
```

```bash
# note the added --snp-list line
# replace ${JOB_ID} with 0, 1, 2, ..., ${{N_JOB - 1}}

admix assoc \
    --pfile toy-admix \
    --pheno toy-admix.pheno \
    --method ATT,TRACTOR \
    --quantile-normalize True \
    --snp-list cache/toy-admix.${JOB_ID}.snplist \
    --out toy-admix
```

### Methodology details
For background, we recommend reading [On powerful GWAS in admixed populations](https://www.nature.com/articles/s41588-021-00953-5). Nature Genetics (2021)
# Overview

Admix-kit provides an integrated toolkit for analyzing admixed genotypes and their relationship with phenotypes. Here we provide an overview of the package.

## Analysis components

We provide an overview of analyses that can be performed using `admix-kit`. The provided total time estimates are based on our experience in analyzing 20K individuals with ~1M genome-wide HapMap3 SNPs. Most of the analyses can be run in parallel in a cluster environment by dividing the genomes into regions (e.g., chromosomes). See each page in details.

| Task                                | Total computing time | Example                               |                          Reference                           |
| ----------------------------------- | :------------------: | ------------------------------------- | :----------------------------------------------------------: |
| Dataset preparation                 |       2 hours        | [Guide](prepare-dataset.md)           |                              -                               |
| Local ancestry inference with RFmix |       5 hours        | [Guide](rfmix.md)                     |                              -                               |
| Genotype simulation                 |       5 hours        | [Guide](simulate-admix-genotype.md)   |                              -                               |
| Association testing                 |       10 hours       | [Notebook](notebooks/assoc.ipynb)     | [CLI](cli/assoc-test.md) / [API](api.md#association-testing) |
| Genetic correlation estimation      |       30 hours       | [Notebook](notebooks/genet-cor.ipynb) |                   [CLI](cli/genet-cor.md)                    |
| Polygenic scoring                   |       2 hours        | [Notebook](notebooks/pgs.ipynb)       |     [CLI](cli/pgs.md) / [API](api.md#polygenic-scoring)      |

## Brief introduction



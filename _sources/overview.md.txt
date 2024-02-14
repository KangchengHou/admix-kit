# Overview

Admix-kit is an integrated toolkit for analyzing admixed genotypes and studying genetic effects of complex traits. Here we provide an overview of the package to help you navigate the different components of this package.


## Navigate the documentation
We prepare an example simulated dataset (see [admixed genotype simulation](simulate-admix-genotype.md)). We provide example workflows for [basic statistics and visualization](notebooks/analyze-admix-simu-data.ipynb), local ancestry-aware [GWAS](notebooks/assoc.ipynb), [genetic correlation estimation](notebooks/genet-cor.ipynb), and [polygenic scoring](notebooks/polygenic-scoring.ipynb). You can find details about [python API](api.md) and [command line interface](cli/index.md).

To perform these analysis on your own dataset, please first follow [data preparation](prepare-dataset.md) steps to format your dataset that admix-kit accepts. Then you can similarly apply the above workflows to your own data.

## Analysis components

We provide an overview of analyses one can perform with `admix-kit`. We provide some total time estimates based on our experience in analyzing 20K individuals with ~1M genome-wide HapMap3 SNPs using a machine with Intel Xeon 2.30GHz CPU and 64GB memory. Most of the analyses can be run in parallel in a cluster environment by dividing the genomes into regions (e.g., chromosomes). See each page in details.

| Task                                | Total computing time | Example                                       |                          Reference                           |
| ----------------------------------- | :------------------: | --------------------------------------------- | :----------------------------------------------------------: |
| Dataset preparation                 |       2 hours        | [Guide](prepare-dataset.md)                   |                              -                               |
| Local ancestry inference with RFmix |       5 hours        | [Guide](rfmix.md)                             |                              -                               |
| Genotype simulation                 |       5 hours        | [Guide](simulate-admix-genotype.md)           |                              -                               |
| Association testing                 |       10 hours       | [Notebook](notebooks/assoc.ipynb)             | [CLI](cli/assoc-test.md) / [API](api.md#association-testing) |
| Genetic correlation estimation      |       30 hours       | [Notebook](notebooks/genet-cor.ipynb)         |                   [CLI](cli/genet-cor.md)                    |
| Polygenic scoring                   |       2 hours        | [Notebook](notebooks/polygenic-scoring.ipynb) |     [CLI](cli/pgs.md) / [API](api.md#polygenic-scoring)      |

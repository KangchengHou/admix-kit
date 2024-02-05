# Overview

Admix-kit provides an integrated toolkit for analyzing admixed genotypes and their relationship with phenotypes. Here we provide an overview of the package.


## Brief introduction
`admix-kit` is designed for analyzing admixed genetic dataset. For genotype-phenotype association testing (admix assoc), we have implemented a suite of methods allowing for genetic effects heterogeneity across local ancestry backgrounds (Pasaniuc et al., 2011; Atkinson et al., 2021; Hou et al., 2021; Mester et al., 2023). These approaches can enhance statistical power by modeling LAS-genetic architecture within admixed populations. We also include functionality to estimate genetic effects concordance across local ancestries (admix genet-cor), which is crucial to understand and inform trait-specific optimal strategies for downstream analyses including polygenic score weight estimation and application (Hou et al., 2023). For polygenic scoring (admix calc-partial-pgs), we implemented calculation of partial polygenic scores, allowing scores to be computed separately for each ancestry background. Such approach can improve polygenic scoring accuracy in admixed populations  (Marnetto et al., 2020; Sun et al., 2022; Cai et al., 2021; Bitarello and Mathieson, 2020).

## Analysis components

We provide an overview of analyses that can be performed using `admix-kit`. The provided total time estimates are based on our experience in analyzing 20K individuals with ~1M genome-wide HapMap3 SNPs using a machine with Intel Xeon 2.30GHz CPU and 64GB memory. Most of the analyses can be run in parallel in a cluster environment by dividing the genomes into regions (e.g., chromosomes). See each page in details.

| Task                                | Total computing time | Example                               |                          Reference                           |
| ----------------------------------- | :------------------: | ------------------------------------- | :----------------------------------------------------------: |
| Dataset preparation                 |       2 hours        | [Guide](prepare-dataset.md)           |                              -                               |
| Local ancestry inference with RFmix |       5 hours        | [Guide](rfmix.md)                     |                              -                               |
| Genotype simulation                 |       5 hours        | [Guide](simulate-admix-genotype.md)   |                              -                               |
| Association testing                 |       10 hours       | [Notebook](notebooks/assoc.ipynb)     | [CLI](cli/assoc-test.md) / [API](api.md#association-testing) |
| Genetic correlation estimation      |       30 hours       | [Notebook](notebooks/genet-cor.ipynb) |                   [CLI](cli/genet-cor.md)                    |
| Polygenic scoring                   |       2 hours        | [Notebook](notebooks/pgs.ipynb)       |     [CLI](cli/pgs.md) / [API](api.md#polygenic-scoring)      |

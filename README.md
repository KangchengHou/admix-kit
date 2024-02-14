# admix-kit: Toolkit for analyzing genetic data of admixed populations
![](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)

This package aims to facilitate analyses and methods development of genetics data from admixed populations. We welcome [feature requests or bug reports](https://github.com/KangchengHou/admix-kit/issues).


## Installation

```bash
pip install -U admix-kit
```

Use `conda` ff you wish to create separate environment to avoid potential conflicts with other packages with:

```bash
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit

conda env create --file environment.yaml
conda activate admix-kit
```

## Get started
- [Overview](https://kangchenghou.github.io/admix-kit/overview.html)
- [Python API](https://kangchenghou.github.io/admix-kit/api.html)
- [Command line interface](https://kangchenghou.github.io/admix-kit/cli/index.html)

## Use cases
The following manuscripts contain more motivation and details for functionalities within `admix-kit`:

1. [Admix-kit: An Integrated Toolkit and Pipeline for Genetic Analyses of Admixed Populations](https://www.biorxiv.org/content/10.1101/2023.09.30.560263v1). bioRxiv 2023
2. [Impact of cross-ancestry genetic architecture on GWASs in admixed populations](https://www.sciencedirect.com/science/article/pii/S0002929723001581). AJHG 2023
3. [Causal effects on complex traits are similar for common variants across segments of different continental ancestries within admixed individuals](https://www.nature.com/articles/s41588-023-01338-6). Nature Genetics  2023
4. [On powerful GWAS in admixed populations](https://www.nature.com/articles/s41588-021-00953-5). Nature Genetics 2021


## Upload to PyPI (for developers)
```
python setup.py sdist
twine upload dist/*
```

<!-- type `PATH=$PATH:~/.local/bin`).  -->

<!-- > To specify a version of admix-kit, use `git clone https://github.com/KangchengHou/admix-kit --branch v0.1`, or replace `v0.1` to other versions listed in https://github.com/KangchengHou/admix-kit/releases. -->
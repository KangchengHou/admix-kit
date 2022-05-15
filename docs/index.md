---
hide-toc: true
---

## admix-kit: Toolkit for analyzing genetic data of admixed populations
`admix-kit` is a Python library to faciliate analyses and methods development of genetics 
data from admixed populations.

This is designed to be a [python library](api.md) to faciliate analyzing admixed populations. 
We have also created [command line interface (CLI)](cli/index.md) for main functionalities.

`admix-kit` can be installed with
```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit
pip install -r requirements.txt; pip install -e .
```

```{note}
`admix-kit` is still in beta version undergoing frequent updates, we welcome any [feedbacks](https://github.com/KangchengHou/admix-kit/pulls) and [bug reports](https://github.com/KangchengHou/admix-kit/issues).   
```

Now, let us jump to quick start with [python](notebooks/quickstart) or [CLI](quickstart-cli.md).

<!-- `admix-kit` has been used in the following projects:
1. On powerful GWAS in admixed populations. [Nature Genetics (2021)](https://www.nature.com/articles/s41588-021-00953-5).
2.  -->


```{toctree}
:hidden:
quickstart-cli
notebooks/quickstart
cli/index
api
```

```{toctree}
:caption: Example usage
:hidden:
prepare-dataset
notebooks/assoc
notebooks/dataset
notebooks/simulate-genotype
notebooks/select-admixed
```
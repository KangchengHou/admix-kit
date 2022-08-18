---
hide-toc: true
---

# admix-kit: Toolkit for analyzing genetic data of admixed populations
`admix-kit` is a Python library to facilitate analyses and methods development of genetics 
data from admixed populations.

This is designed to be a [python library](api.md) to facilitate analyzing admixed populations. 
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

`admix-kit` has been used in the following projects:
1. [On powerful GWAS in admixed populations](https://www.nature.com/articles/s41588-021-00953-5). Nature Genetics (2021).
2. [Causal effects on complex traits are similar across segments of different continental ancestries within admixed individuals](https://www.medrxiv.org/content/10.1101/2022.08.16.22278868v1). medRxiv (2022).


```{toctree}
:hidden:
install
quickstart-cli
notebooks/quickstart
cli/index
api
```

```{toctree}
:caption: Example usage
:hidden:
prepare-dataset
simulate-admix-genotype
notebooks/assoc
notebooks/dataset
marginal-het
notebooks/select-admixed
notebooks/admix-ld
```

<!-- notebooks/simulate-genotype -->

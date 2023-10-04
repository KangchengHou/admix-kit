# admix-kit
![python package](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kangchenghou.github.io/admix-kit)

`admix-kit` is a Python library to facilitate analyses and methods development of genetics data from admixed populations.

> `admix-kit` is currently in beta version and frequently updating. We welcome any [feature requests or bug reports](https://github.com/KangchengHou/admix-kit/issues). Please ask us **"Can `admix-kit` do XX?"**. There is chance that some function has already been implemented but not well documented. We can prioritize more useful work if you let us know.


## Installation
```bash
# Install admix-kit with Python 3.7, 3.8, 3.9
git clone https://github.com/KangchengHou/admix-kit
cd admix-kit && pip install -e .
```
> Installation requires cmake version > 3.12. Use `cmake --version` to check your cmake version. Use `pip install cmake` to install the latest version.

> In some cases, `admix` may not be automatically to your `$PATH` (e.g., because of user installation mode). You need to look through the log and find where `admix` script is installed to and manually add that to your `$PATH` (e.g., type `PATH=$PATH:~/.local/bin`). 

> To specify a version of admix-kit, use `git clone https://github.com/KangchengHou/admix-kit --branch v0.1`, or replace `v0.1` to other versions listed in https://github.com/KangchengHou/admix-kit/releases.

## Get started
- [Full documentation](https://kangchenghou.github.io/admix-kit/index.html)
- [Quick start (Python)](https://kangchenghou.github.io/admix-kit/notebooks/quickstart.html)
- [Application: Genetic correlation across local ancestry](https://kangchenghou.github.io/admix-kit/cli/genet-cor.html)
- [Application: GWAS in admixed populations](https://kangchenghou.github.io/admix-kit/cli/assoc-test.html)

## Use cases
The following manuscripts contain more motivation and details for functionalities within `admix-kit`:

1. Admix-kit: An Integrated Toolkit and Pipeline for Genetic Analyses of Admixed Populations. [bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.09.30.560263v1)
2. Impact of cross-ancestry genetic architecture on GWASs in admixed populations. [AJHG 2023](https://www.sciencedirect.com/science/article/pii/S0002929723001581)
3. Causal effects on complex traits are similar for common variants across segments of different continental ancestries within admixed individuals. [Nature Genetics  2023](https://www.nature.com/articles/s41588-023-01338-6)
4. On powerful GWAS in admixed populations. [Nature Genetics 2021](https://www.nature.com/articles/s41588-021-00953-5)
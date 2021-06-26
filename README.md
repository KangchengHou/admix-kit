# admix-tools
![python package](https://github.com/KangchengHou/admix-tools/actions/workflows/workflow.yml/badge.svg)
![document](https://github.com/KangchengHou/admix-tools/actions/workflows/sphinx.yml/badge.svg)

See [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://kangchenghou.github.io/admix-tools) 
for documentation.

## Install
```
git clone https://github.com/KangchengHou/admix-tools
cd admix-tools
pip install -r requirements.txt; pip install -e .
```

## TODO
- Overview tutorial, add a PCA step and include as covariates.
- Derive F-test in the presence of covariates
- Complete GWAS demonstration and pipelines.
- Add helper function to take vcf file and RFmix local ancestry file and spit out zarr.
- First release: complete pipeline for simulating and estimating genetic correlation.
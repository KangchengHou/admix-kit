# Local ancestry-specific polygenic score 

Calculate PGS for each local ancestry background:

```bash
# the local ancestry should be called in advance and saved to <plink2_prefix>.lanc
# see https://kangchenghou.github.io/admix-kit/prepare-dataset.html#step-2-local-ancestry-inference
# to properly format local ancestry file
# apply --dset-build only when the plink file are not in the same build as weight
admix calc-partial-pgs \
    --plink-path <plink2_prefix>.pgen \
    --weights-path <weight_tsv_path> \
    --dset-build 'hg38->hg19' \
    --out out
```

To also calculate reference PGS for reference populations:

```bash
admix calc-partial-pgs \
    --plink-path <plink2_prefix>.pgen \
    --weights-path <weight_tsv_path> \
    --ref-plink-path 1kg-plink \
    --ref-pops 'CEU,YRI' \
    --dset-build 'hg38->hg19' \
    --out out

# to download the 1,000 Genomes reference plink files, use the following command:
admix get-1kg-ref --dir=1kg-ref --build=hg38
```

## Parameter options
```{eval-rst}
.. autofunction:: admix.cli.calc_partial_pgs
```

## Additional notes


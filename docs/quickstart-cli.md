# Quick start (command line interface)
```bash
# install the admix-kit package, make a new directory and cd into it
# try running the following code

# copy test data
# the test data is built from ASW, CEU and YRI individuals 1,000 Genome projects 
# see tests/test-data/make-toy.sh for scripts to build the toy data
test_data_dir=$(python -c "import admix; print(admix.dataset.get_test_data_dir())")
cp ${test_data_dir}/toy-* ./

# rename the provided .lanc as we are to compute this now
mv toy-admix.lanc toy-admix.old.lanc

# local ancestry inference for ASW individuals with CEU and YRI individuals
admix lanc \
    --pfile toy-admix \
    --ref-pfile toy-all \
    --ref-pop-col "Population" \
    --ref-pops "CEU,YRI" \
    --out toy-admix.lanc

# quantitative phenotype simulation 
# toy-admix.pheno (simulated phenotype) and toy-admix.beta (simulated effects) 
# will be generated
admix simulate-quant-pheno \
    --pfile toy-admix \
    --hsq 0.5 \
    --n-causal 2 \
    --n-sim 2 \
    --seed 1234 \
    --out-prefix toy-admix

# extract PC1, PC2 of the toy-admix.indiv_info as toy-admix.covar
awk '{print $1, $3, $4}' toy-admix.indiv_info > toy-admix.covar

# association testing of individual SNPs of simulated phenotype
# several NA values are expected due to some SNP has zero frequency in this toy dataset
admix assoc \
    --pfile toy-admix \
    --pheno toy-admix.pheno \
    --pheno-col SIM0 \
    --covar toy-admix.covar \
    --method ATT,TRACTOR \
    --out toy-admix.assoc
```
# Guideline to use RFmix for local ancestry inference

```{warning}
This guideline assume you have **phased** genotype in PLINK2 pgen format in hg38 or hg19 coordinates.
PLINK2 can convert many other formats to pgen format [(see here)](https://www.cog-genomics.org/plink/2.0/input). 
Please [raise an issue](https://github.com/KangchengHou/admix-kit/issues) so we can help update this guideline to 
adapt to your data.
```
## Step 1: download and process reference data
```{note}
Before we start, create a new folder to store these reference data (e.g. `/path/to/1kg_pgen`) and
`cd /path/to/1kg_pgen`. This will be the directory where we run the following steps 
(each step will correspond to a script file).
```

### Step 1.1: download 1,000 Genomes reference panel

We download the 1,000 Genomes reference panel from [PLINK2 website](https://www.cog-genomics.org/plink/2.0/resources).
````{tab} hg38
```bash
# step1-download-pgen.sh
# ======================
mkdir -p out/pgen && cd out/pgen
wget https://www.dropbox.com/s/23xlpscis1p5xud/all_hg38_ns.pgen.zst?dl=1 -O hg38.raw.pgen.zst
wget https://www.dropbox.com/s/hy54ba9yvw665xf/all_hg38_ns_noannot.pvar.zst?dl=1 -O hg38.raw.pvar.zst
wget https://www.dropbox.com/s/3j9zg103fi8cjfs/hg38_corrected.psam?dl=1 -O hg38.raw.psam

plink2 --zst-decompress hg38.raw.pgen.zst >hg38.raw.pgen
plink2 --zst-decompress hg38.raw.pvar.zst >hg38.raw.pvar

# related samples
wget https://www.dropbox.com/s/129gx0gl2v7ndg6/deg2_hg38.king.cutoff.out.id?dl=1 -O hg38.king.cutoff.out.id

# basic QC: bi-allelic SNPs, MAC >= 5, chromosome 1-22
plink2 --pfile hg38.raw \
    --allow-extra-chr \
    --rm-dup exclude-all \
    --max-alleles 2 \
    --mac 5 \
    --snps-only \
    --chr 1-22 \
    --set-all-var-ids @:#:\$r:\$a \
    --make-pgen --out hg38

# clean up
rm hg38.*.zst
rm hg38.raw*
cd ../../
```
````

````{tab} hg19
```bash
# step1-download-pgen.sh
# ======================
mkdir -p out/pgen && cd out/pgen
wget https://www.dropbox.com/s/dps1kvlq338ukz8/all_phase3_ns.pgen.zst?dl=1 -O hg19.raw.pgen.zst
wget https://www.dropbox.com/s/uqk3gfhwsvf7bf3/all_phase3_ns_noannot.pvar.zst?dl=1 -O hg19.raw.pvar.zst
wget https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1 -O hg19.raw.psam

plink2 --zst-decompress hg19.raw.pgen.zst >hg19.raw.pgen
plink2 --zst-decompress hg19.raw.pvar.zst >hg19.raw.pvar

# related samples
wget https://www.dropbox.com/s/zj8d14vv9mp6x3c/deg2_phase3.king.cutoff.out.id?dl=1 -O hg19.king.cutoff.out.id

# basic QC: bi-allelic SNPs, MAC >= 5, chromosome 1-22
plink2 --pfile hg19.raw \
    --allow-extra-chr \
    --rm-dup exclude-all \
    --max-alleles 2 \
    --mac 5 \
    --snps-only \
    --chr 1-22 \
    --set-all-var-ids @:#:\$r:\$a \
    --make-pgen --out hg19

# clean up
rm hg19.*.zst
rm hg19.raw*
cd ../../
```
````

### Step 1.2: divide by chromosome to VCF files
This is to export pgen format to VCF files as input to RFmix.

```bash
# step2-divide-chrom.sh
# =====================

# replace hg38 with hg19 as needed
mkdir -p out/vcf
for chrom in {1..22}; do
    plink2 --pfile out/pgen/hg38 \
        --export vcf bgz \
        --chr $chrom \
        --out out/vcf/hg38.chr$chrom
    tabix -p vcf out/vcf/hg38.chr$chrom.vcf.gz
done
```

### Step 1.3: download and process sample map and genetic map
In this step, we download the information for individual in 1,000 Genomes (population information) and
genetic map information.
```python
# step3-download-map.py
# =====================
# save this code into a file and
# call: python3 step3-download-map.py hg38
# replace hg38 with hg19 as needed
import sys
import pandas as pd
import subprocess
import shutil
import os

# ROOT_DIR contains:
# - {ROOT_DIR}/pgen/
# - {ROOT_DIR}/vcf/
# this script will create
# - {ROOT_DIR}/metadata/
ROOT_DIR = "out/"

def process_sample_map(build):
    """Download and format sample map from 1000 Genomes"""

    sample_map = pd.read_csv(
        f"{ROOT_DIR}/pgen/{build}.psam",
        delim_whitespace=True,
    )
    unrelated_id = pd.read_csv(
        f"{ROOT_DIR}/pgen/{build}.king.cutoff.out.id", delim_whitespace=True
    )
    os.makedirs(f"{ROOT_DIR}/metadata", exist_ok=True)
    sample_map[["#IID", "Population", "SuperPop"]].to_csv(
        f"{ROOT_DIR}/metadata/{build}.full_sample.tsv", sep="\t", index=False, header=False
    )
    # filter unrelated
    unrelated_sample_map = sample_map[~sample_map["#IID"].isin(unrelated_id["#IID"])]
    unrelated_sample_map[["#IID", "Population", "SuperPop"]].to_csv(
        f"{ROOT_DIR}/metadata/{build}.unrelated_sample.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    print("Population in unrelated sample map:")
    print(unrelated_sample_map["Population"].value_counts())


def process_genetic_map(build):
    """
    Download and format genetic map
    1. call bash script to download genetic map to out/metadata/genetic_map/raw
    2. process the genetic map and save to out/metadata/genetic_map
    """

    if build == "hg38":
        name = "GRCh38"
    elif build == "hg19":
        name = "GRCh37"
    else:
        raise ValueError("build should be hg38 or hg19")

    cmds = f"""
        mkdir -p {ROOT_DIR}/metadata/genetic_map/raw && cd {ROOT_DIR}/metadata/genetic_map/raw
        wget https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.{name}.map.zip
        unzip plink.{name}.map.zip
    """

    subprocess.check_output(cmds, shell=True)

    for chrom in range(1, 23):
        raw_map = pd.read_csv(
            f"{ROOT_DIR}/metadata/genetic_map/raw/plink.chr{chrom}.{name}.map",
            delim_whitespace=True,
            header=None,
        )
        raw_map = raw_map[[0, 3, 2]]
        raw_map.to_csv(
            f"{ROOT_DIR}/metadata/genetic_map/{build}.chr{chrom}.tsv",
            sep="\t",
            index=False,
            header=False,
        )
    # clean up
    shutil.rmtree(f"{ROOT_DIR}/metadata/genetic_map/raw")

if __name__ == "__main__":
    build = sys.argv[1]
    print("Received build:", build)
    process_sample_map(build)
    process_genetic_map(build)
```

```{note}
After these 3 steps, check if your folder contains the following files (you would also expect some other additional intermediate files; replace `hg38` with `hg19` as needed).
Then we are ready to run RFmix.
```
```
.
├── out
│   ├── metadata
│   │   ├── genetic_map
│   │   │   ├── chr1.tsv
│   │   │   ├── chr2.tsv
│   │   │   ├── ...
│   │   │   ├── chr22.tsv
│   │   ├── hg38.full_sample.tsv
│   │   └── hg38.unrelated_sample.tsv
│   └── vcf
│       ├── hg38.chr1.vcf.gz
│       ├── hg38.chr1.vcf.gz.tbi
│       ├── ...
│       ├── hg38.chr22.vcf.gz
│       ├── hg38.chr22.vcf.gz.tbi
├── step1-download-pgen.sh
├── step2-divide-chrom.sh
└── step3-download-map.py
```

## Step 2: run RFmix

### Step 2.1: download and compile RFmix

Follow `https://github.com/slowkoni/rfmix#building-rfmix` to build the rfmix executables.

### Step 2.2: run RFmix
Now we will run RFmix to infer local ancestry for your genotype file (see how to prepare the genotype file in [Prepare dataset section](prepare-dataset.md)).
```bash
## Specify constants
chrom=XX # this script should be ran 1 chromosome at a time
build=hg38 # replace hg38 with hg19 as needed
REF_DIR=/path/to/1kg/ # see step 1
RFMIX=/path/to/rfmix # see above

pfile=/path/to/your-plink2-file # see "Prepare dataset" section
out_prefix=/path/to/output/chr${chrom} # output prefix

## convert pfile to vcf because RFmix takes vcf as input
plink2 \
    --pfile ${pfile} \
    --chr ${chrom} \
    --output-chr 26 \
    --export vcf bgz \
    --out ${out_prefix}.tmp

tabix -p vcf ${out_prefix}.tmp.vcf.gz

## Filter reference samples
# extract 1st column (sample name) and 2nd column (populations)
# Here, we use CEU, YRI, PEL for reference populations for Latino populations
# NOTE: it is up to your choice what reference populations to use
# e.g., For local ancestry inference of European-African admixed individuals, we 
# recommend using CEU, YRI only.
awk '$2=="CEU" || $2=="YRI" || $2=="PEL" {print $1 "\t" $2}' \
    ${REF_DIR}/metadata/${build}.unrelated_sample.tsv >${out_prefix}.tmp.sample_map.tsv

## run RFmix
${RFMIX} \
    -f ${out_prefix}.tmp.vcf.gz \
    -r ${REF_DIR}/vcf/${build}.chr${chrom}.vcf.gz \
    -m ${out_prefix}.tmp.sample_map.tsv \
    -g ${REF_DIR}/metadata/genetic_map/${build}.chr${chrom}.tsv \
    --chromosome=${chrom} \
    -o ${out_prefix}

## Remove temporary files
# .sis file corresponds to XX
# .fb file corresponds to XX
rm ${out_prefix}*tmp*
rm ${out_prefix}.sis.tsv
rm ${out_prefix}.fb.tsv
```

After you finish this step, you can proceed to the rest of [preparing the dataset](prepare-dataset.md).
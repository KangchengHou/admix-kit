# Guideline to use RFmix for local ancestry inference

```{warning}
This guideline assume you have phased genotype data PLINK2 format in GRCh38 coordinates.
Please [raise an issue](https://github.com/KangchengHou/admix-kit/issues) if you have other data format so we can help update this guideline to 
adapt to your data.
```
## Step 1: process reference data
```{note}
Before we start, create a new folder to store these reference data (e.g. `/path/to/1kg_GRCh38_phased`) and
`cd /path/to/1kg_GRCh38_phased`. This will be the directory where we run the following steps 
(each step will correspond to a script file).
```

<!-- TODO: point downloading 1kg reference data at https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3 -->

### Step 1.1: download 1,000 Genomes reference panel (GRCh38)

In this step, we download 1000 Genomes reference panel (GRCh38). It will take some time (~5 hrs, also depending on network) to download.
```bash
# step1-download-vcf.sh
# =====================
mkdir -p out/vcf
wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/*
```

### Step 1.2: rename chromosome names in vcf files
In this step, we just want to rename the `chr` in `#CHROM` column in the downloaded VCF file.
This step will also take some time (~10 hrs for largest chromosome).

```bash
# step2-rename-chrs.sh
# ====================
# this script is run separately for each chromosome
chrom=XX # fill this with a number (1-22), e.g. provided by cluster task ID ${SGE_TASK_ID}

tmpfile=$(mktemp)

for i in {1..22}; do
    echo "chr$i $i" >>"${tmpfile}"
done

cat "${tmpfile}"

bcftools annotate --rename-chrs "${tmpfile}" \ 
    out/vcf/CCDG_14151_B01_GRM_WGS_2020-08-05_chr"${chrom}".filtered.shapeit2-duohmm-phased.vcf.gz | \ 
    bgzip >out/vcf/chr"${chrom}".nochr.vcf.gz

tabix -p vcf out/vcf/chr"${chrom}".nochr.vcf.gz
```

### Step 1.3: download and process sample map and genetic map
In this step, we download the information for individual in 1,000 Genomes (population information) and
genetic map information.
```python
# step3-download-map.py
# =====================
# run this file as python step3-download-map.py

import pandas as pd
import subprocess

def download_sample_map():
    """Download and format sample map from 1000 Genomes"""

    sample_map = pd.read_csv(
        "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt",
        delim_whitespace=True,
    )
    sample_map[["SampleID", "Population", "Superpopulation"]].to_csv(
        "out/metadata/sample_map.full.tsv", sep="\t", index=False, header=False
    )
    # from https://www.cog-genomics.org/plink/2.0/resources "pedigree-corrected"
    unrelated_sample_map = pd.read_csv(
        "https://www.dropbox.com/s/yozrzsdrwqej63q/phase3_corrected.psam?dl=1",
        delim_whitespace=True,
    )
    assert set(unrelated_sample_map["#IID"]).issubset(set(sample_map["SampleID"]))
    unrelated_sample_map[["#IID", "Population", "SuperPop"]].to_csv(
        "out/metadata/sample_map.unrelated.tsv", sep="\t", index=False, header=False
    )
    print("Population in unrelated sample map:")
    print(unrelated_sample_map["Population"].value_counts())

def download_genetic_map():
    """
    Download and format genetic map
    1. call bash script to download genetic map to out/metadata/genetic_map/raw
    2. process the genetic map and save to out/metadata/genetic_map
    """

    cmds = """
        mkdir -p out/metadata/genetic_map/raw && cd out/metadata/genetic_map/raw
        wget http://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
        unzip plink.GRCh38.map.zip
    """

    subprocess.check_output(cmds, shell=True)

    for chrom in range(1, 23):
        raw_map = pd.read_csv(
            f"out/metadata/genetic_map/raw/plink.chr{chrom}.GRCh38.map",
            delim_whitespace=True,
            header=None,
        )
        raw_map = raw_map[[0, 3, 2]]
        raw_map.to_csv(
            f"out/metadata/genetic_map/chr{chrom}.tsv",
            sep="\t",
            index=False,
            header=False,
        )
if __name__ == "__main__":
    download_sample_map()
    download_genetic_map()
```
```{note}
After these 3 steps, check if your folder contains the following files (you would also expect some other additional intermediate files).
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
│   │   ├── sample_map.full.tsv
│   │   └── sample_map.unrelated.tsv
│   └── vcf
│       ├── chr1.nochr.vcf.gz
│       ├── chr1.nochr.vcf.gz.tbi
│       ├── chr2.nochr.vcf.gz
│       ├── chr2.nochr.vcf.gz.tbi
│       ├── ...
│       ├── chr22.nochr.vcf.gz
│       ├── chr22.nochr.vcf.gz.tbi
├── step1-download-vcf.sh
├── step2-rename-chrs.sh
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
REF_DIR=/path/to/1kg_GRCh38_phased/ # see step 1
RFMIX=/path/to/rfmix # see above

pfile=/path/to/your-plink2-file # see "Prepare dataset" section
out_prefix=/path/to/output/chr${chrom} # output prefix

## convert pfile to vcf because RFmix takes vcf as input
plink2 --pfile ${pfile} \
    --chr ${chrom} \
    --output-chr 26 \
    --export vcf bgz \
    --out ${out_prefix}.tmp

tabix -p vcf ${out_prefix}.tmp.vcf.gz

## Filter reference samples
# extract 1st column (sample name) and 2nd column (populations)
# here we use CEU, YRI, PEL for reference populations for Latino populations
awk '$2=="CEU" || $2=="YRI" || $2=="PEL" {print $1 "\t" $2}' \
    ${REF_DIR}/out/metadata/sample_map.unrelated.tsv >${out_prefix}.tmp.sample_map.tsv

## run RFmix
${RFMIX} \
    -f ${out_prefix}.tmp.vcf.gz \
    -r ${REF_DIR}/out/vcf/chr${chrom}.nochr.vcf.gz \
    -m ${out_prefix}.tmp.sample_map.tsv \
    -g ${REF_DIR}/out/metadata/genetic_map/chr${chrom}.tsv \
    --chromosome=${chrom} \
    -o ${out_prefix}

## Remove temporary files
# .sis file corresponds to XX
# .fb file corresponds to XX
rm ${out_prefix}*tmp*
rm ${out_prefix}.sis.tsv
rm ${out_prefix}.fb.tsv
```
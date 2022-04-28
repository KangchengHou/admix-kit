## Step 1: process reference data

TODO: point downloading 1kg reference data at https://www.cog-genomics.org/plink/2.0/resources#1kg_phase3

We now align the SNPs to a reference panel, e.g. 1000 Genomes project. It is possible
to let the ``bcftools`` directly inferface with the 1000G genotype file on the FTP server
so we don't need to download that (see below). Alternatively, one can download 1000 Genomes 
reference panel (GRCh38) with

.. code-block:: bash

    wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/*


## Step 2: run RFmix

We provide some helper scripts to use RFmix. First follow https://github.com/slowkoni/rfmix#building-rfmix to build the rfmix executables.

One also need to specify the individuals for reference ancestral population. For example,
to specify the file for 1000 Genome reference panel for African American:

.. code-block:: python

    import pandas as pd
    sample_map = pd.read_csv("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt", delim_whitespace=True)
    sample_map[sample_map["Superpopulation"].isin(["EUR", "AFR"])][["SampleID", "Superpopulation"]].to_csv("metadata/sample_map.tsv", sep='\t', index=False, header=False)

One also need to get genetic map specifying the recombination rates. One can download
from online and format it for the purpose.

.. code-block:: bash

    mkdir -p metadata/genetic_map/raw && cd metadata/genetic_map/raw
    wget http://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
    unzip plink.GRCh38.map.zip


.. code-block:: python

    import pandas as pd
    for chrom in range(1, 23):
        raw_map = pd.read_csv(f"metadata/genetic_map/raw/plink.chr{chrom}.GRCh38.map", delim_whitespace=True, header=None)
        raw_map = raw_map[[0, 3, 2]]
        raw_map.to_csv(f"metadata/genetic_map/chr{chrom}.tsv", sep='\t', index=False, header=False)


## Complete code example

Prepare reference data:
```bash

mkdir -p metadata/genetic_map/raw && cd metadata/genetic_map/raw
wget http://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
unzip plink.GRCh38.map.zip
cd ../..
python format_map.py
```
RFmix
```bash
vcf_input=$1 # phased VCF.gz with typed density
chrom=22 # chromosome no. for the vcf file
prefix=$()
rfmix=TODO
vcf_kg=ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_chr${chrom}.filtered.shapeit2-duohmm-phased.vcf.gz


bash align_to_ref.sh ${vcf_input} ${vcf_kg} lanc_input/${prefix}
bash rfmix.sh ${rfmix}
```

```bash    
rfmix=/path/to/rfmix # RFmix executable
chrom=22
sample_vcf=lanc_input/${prefix}.sample.vcf.gz
ref_vcf=lanc_input/${prefix}.ref.vcf.gz
sample_map=metadata/sample_map.tsv
genetic_map=metadata/genetic_map/chr${chrom}.tsv
out_prefix=lanc_output/${prefix}
${rfmix} \
    -f ${sample_vcf} \
    -r ${ref_vcf} \
    --chromosome=${chrom} \
    -m ${sample_map} \
    -g ${genetic_map} \
    -e 1 -n 5 \
    -o ${out_prefix}
```
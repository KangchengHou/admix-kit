Prepare dataset
-------------------

We require a phased vcf file to start the analysis. If the data is genotype array data, 
we recommend using the TOPMed imputation server to obtain both imputation and phasing information.

From the imputed results from TOPMed imputation server, we recommend performing some basic QC:

.. code-block:: bash

    # prefix=/path/to/topmed/chr22, for example
    bcftools filter -i 'INFO/R2>0.8 & INFO/MAF > 0.005' ${prefix}.dose.vcf.gz -Oz -o ${prefix}.impute_qced.vcf.gz
    tabix -p vcf ${prefix}.impute_qced.vcf.gz


One can use LAMP-LD or RFmix to perform local ancestry inference. To prepare the dataset,
for computational efficiency, we recommend to first extract the typed SNPs:

.. code-block:: bash

    bcftools view -i 'TYPED=1|TYPED_ONLY=1' ${prefix}.impute_qced.vcf.gz -Oz -o ${prefix}.typed.vcf.gz
    tabix -p vcf ${prefix}.typed.vcf.gz

.. note::
    You don't have to strictly follow these commands, the goal of these steps is just to
    obtain a phased VCF file with SNPs in typed density to efficiently infer local ancestry.

In the following, we will

1. Align the sample VCF file with reference VCF file (1000 Genome project for example).
2. Infer local ancestry with RFmix


Obtain aligned vcf files
=========================

We now align the SNPs to a reference panel, e.g. 1000 Genomes project. It is possible
to let the ``bcftools`` directly inferface with the 1000G genotype file on the FTP server
so we don't need to download that (see below). Alternatively, one can download 1000 Genomes 
reference panel (GRCh38) with

.. code-block:: bash

    wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/*

We now align the sample VCF file and reference genotype with the following script, for example.

.. code-block:: bash

    chrom=22
    kg_vcf=ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_chr${chrom}.filtered.shapeit2-duohmm-phased.vcf.gz
    bash align_to_ref.sh ${prefix}.typed.vcf.gz ${kg_vcf} lanc_input/${prefix}

where ``align_to_ref.sh`` is as follows:

.. literalinclude:: snippets/align_to_ref.sh
  :language: bash


After this step, you will obtain aligned sample genotype ``${prefix}.sample.vcf.gz`` and 
reference panel genotype ``${prefix}.ref.vcf.gz``

Local ancestry inference with RFmix
===================================

We provide some helper scripts to use RFmix. First follow https://github.com/slowkoni/rfmix#building-rfmix
to build the rfmix executables.

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

Now we are ready to run RFmix

..    rfmix=/u/project/pasaniuc/kangchen/tractor-response/ukb/03_rfmix/rfmix/rfmix
    prefix=chr22

.. code-block:: bash

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


TL;DR
=====

Prepare external reference data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    mkdir -p metadata/genetic_map/raw && cd metadata/genetic_map/raw
    wget http://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh38.map.zip
    unzip plink.GRCh38.map.zip
    cd ../..
    python format_map.py

Perform local ancestry inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    vcf_input=$1 # phased VCF.gz with typed density
    chrom=22 # chromosome no. for the vcf file
    prefix=$()
    rfmix=TODO
    vcf_kg=ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_chr${chrom}.filtered.shapeit2-duohmm-phased.vcf.gz
    
    
    bash align_to_ref.sh ${vcf_input} ${vcf_kg} lanc_input/${prefix}
    bash rfmix.sh ${rfmix}

.. code-block:: bash
    
    
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
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
import warnings
from pandas.api.types import is_string_dtype, is_categorical_dtype
from os.path import dirname, join
from typing import List, Tuple
import os
import scipy
from tqdm import tqdm
import admix


def match_prs_weights(
    dset: xr.Dataset, df_weight: pd.DataFrame, weight_cols: List[str]
) -> Tuple[xr.Dataset, pd.DataFrame]:
    """
    align the dataset and PRS weights with the following 3 steps:
    1. match SNPs in `dset` and `df_weight` by `CHROM` and `POS`
    2. Try match REF and ALT columns in `dset` and `df_weight`, either
        REF_dset = REF_weight, ALT_dset = ALT_weight, or
        REF_dset = ALT_weight, ALT_dset = REF_weight
    3. For mismatched SNPs,all columns in `weight_cols` of returned data frame
        will be reversed

    A shallow copy of `dset` will be returned.
    A copy of `df_weight` will be returned.
    Parameters
    ----------
    dset: a dataset with prs
    df_weight: a dataframe with prs weights
    weight_cols: list of columns in df_weight representing PRS weights
    """

    df_match_dset = (
        dset[["CHROM", "POS", "REF", "ALT"]]
        .to_dataframe()
        .reset_index()
        .rename(columns={"snp": "SNP"})
    )

    df_match_weight = df_weight[["SNP", "CHROM", "POS", "REF", "ALT"]].reset_index()

    df_merged = pd.merge(
        df_match_dset,
        df_match_weight,
        on=["CHROM", "POS"],
        suffixes=["_dset", "_weight"],
    )
    noflip_index = (df_merged["REF_dset"] == df_merged["REF_weight"]) & (
        df_merged["ALT_dset"] == df_merged["ALT_weight"]
    )
    flip_index = (df_merged["REF_dset"] == df_merged["ALT_weight"]) & (
        df_merged["ALT_dset"] == df_merged["REF_weight"]
    )

    rls_df_weight = df_weight.loc[
        df_weight.SNP.isin(df_merged["SNP_weight"][noflip_index | flip_index]),
        ["SNP", "CHROM", "POS", "REF", "ALT"] + weight_cols,
    ].copy()
    rls_df_weight.loc[
        rls_df_weight.SNP.isin(df_merged["SNP_weight"][flip_index]), weight_cols
    ] *= -1

    # the following step can be a bit slow sometimes, need to make sure both array
    # are like <U18 types to be fast
    rls_dset = dset.sel(
        snp=np.isin(
            dset.snp.values,
            df_merged["SNP_dset"][noflip_index | flip_index].values.astype(str),
        )
    )
    return rls_dset, rls_df_weight


def impute_lanc(dset: xr.Dataset, dset_ref: xr.Dataset):
    """
    Impute local ancestry using a reference dataset. The two data sets are assumed to
    have the same haplotype order, etc. Typically they are just a subset of each other.

    Using the following steps:
        1. basic checks are performed for the two data sets.
        2. `dset_ref`'s individuals is matched with `dset`, `dset`'s individuals
            must be a subset of `dset_ref`'s individuals.
        3. Imputation are performed

    Parameters
    ----------
    dset: a data set to be imputed with local ancestry
    dset_ref: a data set with local ancestry for reference

    Returns
    -------
    dset_imputed: a data set with imputed local ancestry
    """
    assert (
        len(set(dset.coords["CHROM"].data)) == 1
    ), "Data set to be imputed can only have one chromosome"

    # dset.indiv is a subset of dset_ref.indiv
    assert set(dset.indiv.values) <= set(dset_ref.indiv.values)
    # align the individuals order for two data sets
    dset_ref = dset_ref.sel(indiv=dset.indiv.values).sel(
        snp=(dset_ref.coords["CHROM"] == dset.coords["CHROM"][0])
    )

    # find relevant regions in reference dataset with local ancestry (hapmap3 SNPs here)
    ref_start = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[0]))
    ref_stop = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[-1]))

    dset_ref = dset_ref.isel(snp=slice(ref_start, ref_stop + 1))

    imputed_lanc = []
    for ploidy_i in range(2):
        # form a dataframe which contains the known local ancestry and locations to be imputed
        df_snp = pd.concat(
            [dset.snp.to_dataframe()[["POS"]], dset_ref.snp.to_dataframe()[["POS"]]]
        )
        df_snp = df_snp[~df_snp.index.duplicated()].sort_values("POS")

        df_lanc = pd.DataFrame(
            index=df_snp.index.values, columns=dset["indiv"].values, dtype=float
        )
        # fill margin
        df_lanc.iloc[0, :] = dset_ref.lanc[:, 0, ploidy_i]
        df_lanc.iloc[-1, :] = dset_ref.lanc[:, -1, ploidy_i]

        # fill inside
        df_lanc.loc[dset_ref.snp.values, :] = dset_ref["lanc"][:, :, ploidy_i].values.T

        # interpolate
        df_lanc = df_lanc.reset_index().interpolate(method="nearest").set_index("index")

        imputed_lanc.append(df_lanc.loc[dset["snp"].values, :].values.astype(np.int8).T)

    # imputed_lanc is in the order of ("indiv", "snp", "ploidy")
    # determine the dim order from dset.geno

    dset = dset.assign(
        lanc=(
            ("indiv", "snp", "ploidy"),
            da.from_array(np.dstack(imputed_lanc), chunks=-1),
        )
    )

    return dset


def impute_lanc_old(dset: xr.Dataset, dset_ref: xr.Dataset):
    """
    Impute local ancestry using a reference dataset. The two data sets are assumed to
    have the same haplotype order, etc. Typically they are just a subset of each other.

    Using the following steps:
        1. basic checks are performed for the two data sets.
        2. `dset_ref`'s individuals is matched with `dset`, `dset`'s individuals
            must be a subset of `dset_ref`'s individuals.
        3. Imputation are performed

    Parameters
    ----------
    dset: a data set to be imputed with local ancestry
    dset_ref: a data set with local ancestry for reference

    Returns
    -------
    dset_imputed: a data set with imputed local ancestry
    """
    assert (
        len(set(dset.coords["CHROM"].data)) == 1
    ), "Data set to be imputed can only have one chromosome"

    # dset.indiv is a subset of dset_ref.indiv
    assert set(dset.indiv.values) <= set(dset_ref.indiv.values)
    dset_ref = dset_ref.sel(indiv=dset.indiv.values).sel(
        snp=(dset_ref.coords["CHROM"] == dset.coords["CHROM"][0])
    )

    # find relevant regions in reference dataset with local ancestry (hapmap3 SNPs here)
    # ref_start = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[0]))
    ref_start = np.where(dset_ref["POS"] < dset["POS"][0])[0][-1]
    # ref_stop = np.argmin(np.abs(dset_ref["POS"].values - dset["POS"].values[-1]))
    ref_stop = np.where(dset_ref["POS"] > dset["POS"][-1])[0][0]

    dset_ref_subset = dset_ref.isel(snp=slice(ref_start, ref_stop + 1))
    dset_ref_margin = dset_ref_subset.isel(snp=[0, -1])

    imputed_lanc = []
    for ploidy_i in range(2):
        df_ref_margin = pd.DataFrame(
            dset_ref_margin.lanc[:, ploidy_i].values.T,
            columns=dset_ref_margin.indiv.values,
            index=dset_ref_margin.snp.values,
        )
        df_imputed = pd.DataFrame(
            {
                "snp": dset.snp["snp"],
            }
        ).set_index("snp")
        df_imputed = pd.concat(
            [
                df_imputed,
                pd.DataFrame(columns=dset["indiv"].values, dtype=float),
            ]
        )
        # fill margin
        df_imputed = pd.concat(
            [df_ref_margin.iloc[[0], :], df_imputed, df_ref_margin.iloc[[-1], :]],
            axis=0,
        )
        df_imputed.index.name = "snp"
        # fill inside
        df_imputed.loc[
            dset_ref_subset.snp.values, dset_ref_subset.indiv.values
        ] = dset_ref_subset["lanc"][:, :, ploidy_i].values.T
        # interpolate
        df_imputed = (
            df_imputed.reset_index().interpolate(method="nearest").set_index("snp")
        )

        imputed_lanc.append(
            df_imputed.loc[dset["snp"].values, dset["indiv"].values]
            .values.astype(np.int8)
            .T
        )

    dset = dset.assign(
        lanc=(
            ("indiv", "snp", "ploidy"),
            da.from_array(np.dstack(imputed_lanc), chunks=-1),
        )
    )

    return dset


def impute_std(geno, mean=None, std=None):
    """
    impute the mean and then standardize
    geno: (num_indivs x num_snps) numpy array
    """
    if mean is None and std is None:
        mean = np.nanmean(geno, axis=0)
        nanidx = np.where(np.isnan(geno))
        geno[nanidx] = mean[nanidx[1]]
        std = np.std(geno, axis=0)
        std_geno = (geno - mean) / std
    else:
        nanidx = np.where(np.isnan(geno))
        geno[nanidx] = mean[nanidx[1]]
        std_geno = (geno - mean) / std
    return std_geno


def mean_std(geno, chunk_size=500):
    row_count = geno.row_count
    col_count = geno.col_count
    mean = np.zeros(col_count)
    std = np.zeros(col_count)

    for i in range(0, col_count, chunk_size):
        sub_geno = geno[:, i : i + chunk_size].read().val
        sub_mean = np.nanmean(sub_geno, axis=0)
        mean[i : i + chunk_size] = sub_mean
        nanidx = np.where(np.isnan(sub_geno))
        sub_geno[nanidx] = sub_mean[nanidx[1]]
        std[i : i + chunk_size] = np.std(sub_geno, axis=0)
    df = pd.DataFrame({"mean": mean, "std": std})
    return df


def cov(geno, mean_std, chunk_size=500):
    """
    geno: a Bed object which we are interested in estimating the LD
    chunk_size: number of SNPs to compute at a time
    """
    # first compute the mean and standard deviation for the given geno
    num_indv, num_snps = geno.shape
    cov = np.zeros([num_snps, num_snps])
    # compute mean and standard deviation
    for row_start in range(0, num_snps, chunk_size):
        for col_start in range(0, num_snps, chunk_size):
            # for each block
            row_stop = row_start + chunk_size
            col_stop = col_start + chunk_size
            if row_stop > num_snps:
                row_stop = num_snps
            if col_stop > num_snps:
                col_stop = num_snps

            std_row_geno = impute_std(
                geno[:, row_start:row_stop].read().val,
                mean_std["mean"][row_start:row_stop].values,
                mean_std["std"][row_start:row_stop].values,
            )
            std_col_geno = impute_std(
                geno[:, col_start:col_stop].read().val,
                mean_std["mean"][col_start:col_stop].values,
                mean_std["std"][col_start:col_stop].values,
            )

            cov[
                np.ix_(np.arange(row_start, row_stop), np.arange(col_start, col_stop))
            ] = np.dot(std_row_geno.T, std_col_geno) / (std_row_geno.shape[0])
    return cov


def quad_form(x, A):
    return np.dot(np.dot(x.T, A), x)


def zsc2pval(zsc):
    return 1 - scipy.stats.norm.cdf(zsc)


def pval2zsc(pval):
    return -scipy.stats.norm.ppf(pval)


def chi2_to_logpval(chi2, dof=1):
    return scipy.stats.chi2.logsf(chi2, dof)


def check_align(dsets: List[xr.Dataset], dim: str) -> bool:
    """takes 2 or more datasets, and check whether attributes align

    Parameters
    ----------
    dsets : List[xr.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    bool: whether the two datasets align in the given dimension
    """
    assert dim in ["snp", "indiv"]
    return False


def align(dsets: List[xr.Dataset], dim: str) -> List[xr.Dataset]:
    """takes 2 or more datasets, return the aligned dataset

    Parameters
    ----------
    dsets : List[xr.Dataset]
        List of datasets
    dim : str
        which dimension to check

    Returns
    -------
    List[xr.Dataset]: list of aligned datasets
    """
    assert dim in ["snp", "indiv"]
    pass

    return


def make_dataset(
    geno, snp: pd.DataFrame, indiv: pd.DataFrame, meta: dict = None, lanc=None
):
    """Make up a dataset

    Parameters
    ----------
    geno : (#indiv, #snp, 2) array_like
        Genotype count matrix
    snp : pd.DataFrame
        Index is identifier to SNP
        each column corresponds to an attribute
    indiv : pd.DataFrame
        Index is identifier to individuals
        each column corresponds to an attribute
    meta : dict, optional
        Meta information about the dataset
    lanc : (#indiv, #snp, 2) array_like, optional
        Local ancestry
    """
    # fill `data_vars`
    data_vars: dict = {
        "geno": (("indiv", "snp", "ploidy"), geno),
    }
    if lanc is not None:
        data_vars["lanc"] = (("indiv", "snp", "ploidy"), lanc)

    coords: dict = {}
    # fill SNP information
    coords["snp"] = snp.index.values
    if not is_string_dtype(coords["snp"]):
        warnings.warn("Transforming snp index to str")
    coords["snp"] = coords["snp"].astype(str)

    for col in snp.columns:
        vals = snp[col].values
        if is_string_dtype(snp[col]):
            vals = snp[col].values.astype(str)

        coords[col] = ("snp", vals)

    # fill in individual information
    coords["indiv"] = indiv.index.values
    if not is_string_dtype(coords["indiv"]):
        warnings.warn("Transforming indiv index to str")
    coords["indiv"] = coords["indiv"].astype(str)

    for col in indiv.columns:
        vals = indiv[col].values
        if is_string_dtype(indiv[col]):
            vals = vals.astype(str)
        coords[col] = ("indiv", vals)

    dset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=meta)
    return dset


def seperate_ld_blocks(anc, phgeno, legend, ld_blocks):
    assert len(legend) == anc.shape[1]
    assert len(legend) == phgeno.shape[1]

    rls_list = []
    for block_i, block in ld_blocks.iterrows():
        block_index = np.where(
            (block.START <= legend.position) & (legend.position < block.STOP)
        )[0]
        block_legend = legend.loc[block_index]
        block_anc = anc[:, block_index]
        block_phgeno = phgeno[:, block_index]
        rls_list.append((block_anc, block_phgeno, block_legend))
    return rls_list


"""
load compiled data sets
"""


# PAGE admixed individuals with EUR - AFR ancestries in hapmap3 density
def load_page_eur_afr_hm3(
    chrom=None,
    GENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s03_aframr/dataset/hm3.zarr/",
    PHENO_DIR="/u/project/sgss/PAGE/phenotype/",
):
    if chrom is None:
        chrom = np.arange(1, 23)
    elif isinstance(chrom, List):
        chrom = np.array(chrom)
    elif isinstance(chrom, int):
        chrom = np.array([chrom])
    else:
        raise ValueError("chrom must be None, List or int")

    trait_cols = [
        # Inflammtory traits
        "crp",
        "total_wbc_cnt",
        "mean_corp_hgb_conc",
        "platelet_cnt",
        # lipid traits
        "hdl",
        "ldl",
        "triglycerides",
        "total_cholesterol",
        # lifestyle traits
        "cigs_per_day_excl_nonsmk_updated",
        "coffee_cup_day",
        # glycemic traits
        "a1c",
        "insulin",
        "glucose",
        "t2d_status",
        # electrocardiogram traits
        "qt_interval",
        "qrs_interval",
        "pr_interval",
        # blood pressure traits
        "systolic_bp",
        "diastolic_bp",
        "hypertension",
        # anthropometric traits
        "waist_hip_ratio",
        "height",
        "bmi",
        # kidney traits
        "egfrckdepi",
    ]

    covar_cols = ["study", "age", "sex", "race_ethnicity", "center"] + [
        f"geno_EV{i}" for i in range(1, 51)
    ]

    race_encoding = {
        1: "Unclassified",
        2: "African American",
        3: "Hispanic/Latino",
        4: "Asian",
        5: "Native Hawaiian",
        6: "Native American",
        7: "Other",
    }

    race_color = {
        "Unclassified": "#ffffff",
        "African American": "#e9d16a",
        "Hispanic/Latino": "#9a3525",
        "Asian": "#3c859d",
        "Native Hawaiian": "#959f6e",
        "Native American": "#546f7b",
        "Other": "#d07641",
    }

    df_pheno = pd.read_csv(
        join(PHENO_DIR, "MEGA_page-harmonized-phenotypes-pca-freeze2-2016-12-14.txt"),
        sep="\t",
        na_values=".",
        low_memory=False,
    )
    df_pheno["race_ethnicity"] = df_pheno["race_ethnicity"].map(race_encoding)

    dset_list = []
    for i_chr in tqdm(chrom):
        dset_list.append(xr.open_zarr(join(GENO_DIR, f"chr{i_chr}.zip")))

    dset = xr.concat(dset_list, dim="snp")

    df_aframr_pheno = df_pheno.set_index("PAGE_Subject_ID").loc[
        dset.indiv.values, trait_cols + covar_cols
    ]

    for col in df_aframr_pheno.columns:
        dset[f"{col}@indiv"] = ("indiv", df_aframr_pheno[col].values)

    for col in ["center", "study", "race_ethnicity"]:
        dset[f"{col}@indiv"] = dset[f"{col}@indiv"].astype(str)

    # format the dataset to follow the new standards
    for k in dset.data_vars.keys():
        if k.endswith("@indiv"):
            dset.coords[k.split("@")[0]] = ("indiv", dset.data_vars[k].data)
        if k.endswith("@snp"):
            dset.coords[k.split("@")[0]] = ("snp", dset.data_vars[k].data)
    dset = dset.drop_vars(
        [
            k
            for k in dset.data_vars.keys()
            if (k.endswith("@indiv") or k.endswith("@snp"))
        ]
    )

    dset = dset.rename({n: n.split("@")[0] for n in [k for k in dset.coords.keys()]})

    return dset


def load_page_eur_afr_imputed(
    region: str,
    imputed_vcf_dir: str = "/u/project/pasaniuc/pasaniucdata/admixture/projects/PAGE-QC/s01_dataset/all",
):
    """
    PAGE admixed individuals with EUR - AFR ancestries in imputed density

    Parameters
    ----------
    region : str
        Region name, in the format of "chrom:start-stop", e.g. "1:1-100000"
    imputed_vcf_dir : str
        Directory of imputed VCF files
    """
    chrom = int(region.split(":")[0])
    start, stop = [int(i) for i in region.split(":")[1].split("-")]
    print(chrom)
    # load hm3 data set
    dset_hm3 = load_page_eur_afr_hm3(chrom=chrom)

    dset = admix.io.read_vcf(
        join(imputed_vcf_dir, f"chr{chrom}.vcf.gz"),
        region=f"{chrom}:{start}-{stop}",
    ).sel(indiv=dset_hm3.indiv.values)

    dset = admix.data.impute_lanc(dset=dset, dset_ref=dset_hm3)
    dset.attrs["n_anc"] = 2
    return dset


# UK Biobank admixed individuals with EUR - AFR ancestries in hapmap3 density
def load_ukb_eur_afr_hm3(
    chrom=None,
    GENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/admix-analysis/dataset",
    PHENO_DIR="/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/pheno",
):
    if chrom is None:
        chrom = np.arange(1, 23)
    elif isinstance(chrom, List):
        chrom = np.array(chrom)
    elif isinstance(chrom, int):
        chrom = np.array([chrom])
    else:
        raise ValueError("chrom must be None, List or int")

    dset_list = []
    for i_chr in tqdm(chrom):
        dset_list.append(xr.open_zarr(join(GENO_DIR, f"chr{i_chr}.zarr")))

    dset = xr.concat(dset_list, dim="snp")

    trait_cols = ["height"]

    for trait in trait_cols:
        # raw phenotype
        df_pheno = pd.read_csv(
            join(PHENO_DIR, f"admix.{trait}.pheno"),
            delim_whitespace=True,
            low_memory=False,
        )

        df_pheno.index = df_pheno.FID.astype(str) + "_" + df_pheno.IID.astype(str)
        df_pheno = df_pheno.drop(columns=["FID", "IID"])
        dset.coords[trait] = "indiv", df_pheno[trait].reindex(dset.indiv.values)

        # residual phenotype
        df_pheno = pd.read_csv(
            join(PHENO_DIR, f"admix.{trait}.residual_pheno"),
            delim_whitespace=True,
            low_memory=False,
        )

        df_pheno.index = df_pheno.FID.astype(str) + "_" + df_pheno.IID.astype(str)
        df_pheno = df_pheno.drop(columns=["FID", "IID"])
        dset.coords[f"{trait}_residual"] = (
            "indiv",
            df_pheno[f"{trait}-residual"].reindex(dset.indiv.values),
        )

    df_covar = pd.read_csv(
        join(PHENO_DIR, f"admix.covar"),
        delim_whitespace=True,
        low_memory=False,
    )

    df_covar.index = df_covar.FID.astype(str) + "_" + df_covar.IID.astype(str)
    df_covar = df_covar.drop(columns=["FID", "IID"])

    for col in df_covar:
        dset.coords[col] = "indiv", df_covar[col].reindex(dset.indiv.values)

    dset = dset.rename({n: n.split("@")[0] for n in [k for k in dset.coords.keys()]})

    return dset


def load_ukb_eur_afr_imputed(
    region: str,
    imputed_vcf_dir: str = "/u/project/pasaniuc/pasaniucdata/admixture/projects/admix-prs-uncertainty/data/PLINK/admix/topmed",
):
    """
    UKB admixed individuals with EUR - AFR ancestries in imputed density

    Parameters
    ----------
    region : str
        Region name, in the format of "chrom:start-stop", e.g. "1:1-100000"
    imputed_vcf_dir : str
        Directory of imputed VCF files
    """
    chrom = int(region.split(":")[0])
    start, stop = [int(i) for i in region.split(":")[1].split("-")]
    # load hm3 data set
    dset_hm3 = load_ukb_eur_afr_hm3(chrom=chrom)

    dset = admix.io.read_vcf(
        join(imputed_vcf_dir, f"chr{chrom}/chr{chrom}.sample.imputed.vcf.gz"),
        region=f"chr{chrom}:{start}-{stop}",
    )
    if dset is None:
        warnings.warn(f"No SNPs found in {region}, `None` is returned")
        return None

    dset = dset.sel(indiv=dset_hm3.indiv.values)

    # impute local ancestry
    dset = admix.data.impute_lanc(dset=dset, dset_ref=dset_hm3)
    dset.attrs["n_anc"] = 2
    return dset


def load_lab_dataset(name: str, chrom: int = None, region: str = None) -> xr.Dataset:
    """Load prepared dataset in Bogdan lab, currently available
    if you use this function on cluster, and have the proper access to the data.
    Contact kangchenghou@gmail.com before using this function.

    Parameters
    ----------
    name : str
        Name of the dataset
        - `simulate_eur_afr.20_80`: Simulated admixture of 20% EUR and 80% AFR
        - `simulate_eur_afr.50_50`: Simulated admixture of 50% EUR and 50% AFR
        - `ukb_eur_afr_hm3`: Admixed individuals of European and African ancestries
            in UK Biobank with hapmap3 SNPs
        - `ukb_eur_afr_imputed`: Admixed individuals of European and African ancestries
            in UK Biobank with imputed SNPs
        - `page_eur_afr_hm3`: Admixed individuals of European and African ancestries
            in PAGE with hapmap3 SNPs
        - `page_eur_afr_imputed`: Admixed individuals of European and African ancestries
            in PAGE with imputed SNPs
    chrom : int, optional
        Chromosome number
    region : str, optional
        Region name

    Returns
    -------
    xr.Dataset
        dataset
    """
    if name.startswith("simulate_eur_afr"):
        assert name in [
            "simulate_eur_afr.20_80",
            "simulate_eur_afr.50_50",
        ]
        root_dir = "/u/project/pasaniuc/pasaniucdata/admixture/dataset"
        assert os.path.isdir(root_dir), f"check that you have access to {root_dir}"
        dset = xr.open_zarr(join(root_dir, name + ".zip"))
    elif name == "ukb_eur_afr_hm3":
        assert region is None, "region must not be specified when loading hm3 data"
        dset = load_ukb_eur_afr_hm3(chrom=chrom)
    elif name == "ukb_eur_afr_imputed":
        assert region is not None, "region must be specified when loading imputed data"
        assert chrom is None, "chrom must not be specified when loading imputed data"
        dset = load_ukb_eur_afr_imputed(region=region)
    elif name == "page_eur_afr_hm3":
        assert region is None, "region must not be specified when loading hm3 data"
        dset = load_page_eur_afr_hm3(chrom=chrom)
    elif name == "page_eur_afr_imputed":
        assert region is not None, "region must be specified when loading imputed data"
        assert chrom is None, "chrom must not be specified when loading imputed data"
        dset = load_page_eur_afr_imputed(region=region)
    return dset
import admix
import pandas as pd
from typing import Union, List, Optional
from admix import logger
from ._utils import log_params


def assoc(
    pfile: str,
    pheno: str,
    out: str,
    pheno_col: Optional[str] = None,
    covar: Optional[str] = None,
    method: Union[str, List[str]] = "ATT",
    family: str = "quant",
    quantile_normalize: bool = False,
    snp_list: str = None,
    fast: bool = True,
):
    """
    Perform association testing.

    Parameters
    ----------
    pfile : str
        Prefix to the PLINK2 file (.pgen should not be added). If method that requires
        local ancestry is specified, a matched :code:`<pfile>.lanc` file should exist.
    pheno : str
        Path to the phenotype file. The text file should be space delimited with header 
        and one individual per row. The first column should be the individual ID. Use 
        :code:`--pheno-col` to specify the column for the phenotype value 
    pheno_col : Optional[str]
        Column name for the phenotype value. NaN should be encoded as "NA" and these 
        individuals will be removed in the analysis. Binary phenotype should be encoded 
        as 0 and 1, and :code:`--family binary` should be used.
    covar: Optional[str]
        Path to the covariate file. The text file should be space delimited with header 
        and one individual per row. The first column should be the individual ID, and 
        the remaining columns should be the covariate values. All columns will be used
        for the analysis. NaN should be encoded as "NA" and NaN will be imputed with 
        the mean of each covariate. Categorical covariates will be converted to one
        hot encodings internally.
    out : str
        Path the output file. <out>.<method>.assoc will be created. 
    method : Union[str, List[str]]
        Method to use for association analysis (default ATT). Other methods include:
        TRACTOR, ADM, SNP1, HET
    family : str
        Family to use for association analysis (default quant). One of :code:`quant` or 
        :code:`binary`.
    quantile_normalize : bool
        Whether to quantile normalize the phenotype and every covariate.
    snp_list : str
        Path to the SNP list file. Each line should be a SNP ID. Only SNPs in the
        list will be used for the analysis.
    fast : bool
        Whether to use fast mode (default True).

    Examples
    --------
    .. code-block:: bash

        # See complete example at 
        # https://kangchenghou.github.io/admix-kit/quickstart-cli.html
        admix assoc \\
            --pfile toy-admix \\
            --pheno toy-admix.pheno \\
            --pheno-col SIM0 \\
            --covar toy-admix.covar \\
            --method ATT,TRACTOR \\
            --quantile-normalize True \\
            --out toy-admix
    """
    log_params("assoc", locals())
    assert family in ["quant", "binary"], "family must be either quant or binary"

    # TODO: infer block size using memory use in dask-pgen read
    dset = admix.io.read_dataset(pfile)
    admix.logger.info(f"{dset.n_snp} SNPs and {dset.n_indiv} individuals are loaded")
    df_pheno = pd.read_csv(pheno, delim_whitespace=True, index_col=0, low_memory=False)
    if pheno_col is None:
        pheno_col = df_pheno.columns[0]

    df_pheno = df_pheno[[pheno_col]]
    dset.append_indiv_info(df_pheno, force_update=True)
    # adding covariates
    if covar is not None:
        df_covar = pd.read_csv(
            covar, delim_whitespace=True, index_col=0, low_memory=False
        )
        dset.append_indiv_info(df_covar, force_update=True)
        covar_cols = df_covar.columns
        admix.logger.info(
            f"{len(covar_cols)} covariates are loaded: {','.join(covar_cols)}"
        )
    else:
        covar_cols = None
        admix.logger.info("No covariates are loaded")
    # retain only individuals with non-missing phenotype,
    # or with non-completely missing covariate
    indiv_mask = ~dset.indiv[pheno_col].isna().values
    if covar_cols is not None:
        covar_mask = ~(dset.indiv[covar_cols].isna().values.all(axis=1))
        indiv_mask &= covar_mask
    dset = dset[:, indiv_mask]
    admix.logger.info(
        f"{dset.n_indiv} individuals left "
        "after filtering for missing phenotype, or completely missing covariate"
    )

    # filter for SNPs
    if snp_list is not None:
        with open(snp_list, "r") as f:
            filter_snp_list = [line.strip() for line in f]
        n_filter_snp = len(filter_snp_list)
        filter_snp_list = [snp for snp in filter_snp_list if snp in dset.snp.index]
        if len(filter_snp_list) < n_filter_snp:
            admix.logger.warning(
                f"{n_filter_snp - len(filter_snp_list)} SNPs in {snp_list} are not in the dataset"
            )
        dset = dset[filter_snp_list]

    admix.logger.info(
        f"{dset.n_snp} SNPs and {dset.n_indiv} individuals in the analysis"
    )

    if isinstance(method, str):
        method = [method]

    # extract pheno_values and covar_values
    pheno_values = dset.indiv[pheno_col].values
    if covar_cols is not None:
        covar_values = admix.data.convert_dummy(dset.indiv[covar_cols]).values
    else:
        covar_values = None

    if quantile_normalize:
        pheno_values = admix.data.quantile_normalize(pheno_values)
        for i in range(covar_values.shape[1]):
            covar_values[:, i] = admix.data.quantile_normalize(covar_values[:, i])

    dict_rls = {}
    for m in method:
        admix.logger.info(f"Performing association analysis with method {m}")
        dict_rls[m] = admix.assoc.marginal(
            dset=dset,
            pheno=pheno_values,
            cov=covar_values,
            method=m,
            family="logistic" if family == "binary" else "linear",
            fast=fast,
        )
    for m in dict_rls:
        dict_rls[m].to_csv(
            f"{out}.{m}.assoc", sep="\t", float_format="%.6g", na_rep="NA"
        )
        logger.info(f"Output written to {out}.{m}.assoc")
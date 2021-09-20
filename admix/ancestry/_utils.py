import numpy as np
import pandas as pd
import xarray as xr


def impute_lanc(vcf, ds):
    """Impute a dataset with local ancestry

    Parameters
    ----------
    vcf : [type]
        [description]
    ds : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    gt = vcf["calldata/GT"]
    assert (gt == -1).sum() == 0

    ds_imputed = xr.Dataset(
        data_vars={
            "geno": (("indiv", "snp", "ploidy"), np.swapaxes(gt, 0, 1)),
        },
        coords={
            "snp": vcf["variants/ID"],
            "indiv": vcf["samples"],
            "CHROM": ("snp", vcf["variants/CHROM"]),
            "POS": ("snp", vcf["variants/POS"]),
            "REF": ("snp", vcf["variants/REF"]),
            "ALT": ("snp", vcf["variants/ALT"][:, 0]),
        },
        attrs={"n_anc": 2},
    )

    # fill in individual information
    dict_indiv = {}
    for col in ds["indiv"].coords:
        if col == "indiv":
            assert np.all(ds["indiv"] == ds_imputed["indiv"])
        else:
            dict_indiv[col] = ("indiv", ds[col])
    ds_imputed = ds_imputed.assign_coords(dict_indiv)

    # impute local ancestry

    # relevant typed region
    typed_start = np.where(ds["POS"] < vcf["variants/POS"][0])[0][-1]
    typed_stop = np.where(ds["POS"] > vcf["variants/POS"][-1])[0][0]
    ds_typed_subset = ds.isel(snp=slice(typed_start, typed_stop + 1))
    ds_typed_margin = ds_typed_subset.isel(snp=[0, -1])

    imputed_lanc = []
    for i_hap in range(2):
        df_typed_margin = pd.DataFrame(
            ds_typed_margin.lanc[:, i_hap].values.T,
            columns=ds_typed_margin.indiv.values,
            index=ds_typed_margin.snp.values,
        )
        df_imputed = pd.DataFrame(
            {
                "snp": ds_imputed.snp["snp"],
            }
        ).set_index("snp")
        df_imputed = pd.concat(
            [df_imputed, pd.DataFrame(columns=ds_imputed["indiv"].values, dtype=float)]
        )
        # fill margin
        df_imputed = pd.concat(
            [df_typed_margin.iloc[[0], :], df_imputed, df_typed_margin.iloc[[-1], :]],
            axis=0,
        )
        df_imputed.index.name = "snp"
        # fill inside
        df_imputed.loc[
            ds_typed_subset.snp.values, ds_typed_subset.indiv.values
        ] = ds_typed_subset["lanc"][:, :, i_hap].values.T
        # interpolate
        df_imputed = (
            df_imputed.reset_index().interpolate(method="nearest").set_index("snp")
        )

        imputed_lanc.append(
            df_imputed.loc[ds_imputed["snp"].values, ds_imputed["indiv"].values]
            .values.astype(np.int8)
            .T
        )

    ds_imputed = ds_imputed.assign(
        lanc=(("indiv", "snp", "ploidy"), np.dstack(imputed_lanc))
    )
    return ds_imputed

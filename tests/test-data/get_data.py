import numpy as np
import pandas as pd
from admix.data import read_digit_mat
import xarray as xr

dict_url = {
    "admix_hap": "https://www.dropbox.com/s/xobrfqw8qa6pry0/admix.hap?dl=1",
    "admix_lanc": "https://www.dropbox.com/s/bjuhcjtyfsau927/admix.lanc?dl=1",
    "afr_hap": "https://www.dropbox.com/s/efyltlzksplpxud/afr.hap?dl=1",
    "eur_hap": "https://www.dropbox.com/s/dr2bjdh9b63pqad/eur.hap?dl=1",
    "legend": "https://www.dropbox.com/s/0vlhxvs6lv7r2cp/legend.csv?dl=1",
}

admix_hap = read_digit_mat(dict_url["admix_hap"])
admix_lanc = read_digit_mat(dict_url["admix_lanc"])
afr_hap = read_digit_mat(dict_url["afr_hap"]).T
eur_hap = read_digit_mat(dict_url["eur_hap"]).T
legend = pd.read_csv(dict_url["legend"])

print("admix_hap: ", admix_hap.shape)
print("eur_hap: ", eur_hap.shape)
print("afr_hap: ", afr_hap.shape)

n_anc_haplo = 100
n_admix_haplo = 100

n_snp = admix_lanc.shape[1]


def get_subset(hap, n_snp, n_haplo):
    return np.dstack(
        (
            hap[:, 0:n_snp][np.arange(0, n_haplo, 2)],
            hap[:, 0:n_snp][np.arange(1, n_haplo, 2)],
        )
    )


admix_hap = get_subset(admix_hap, n_snp, n_admix_haplo)
admix_lanc = get_subset(admix_lanc, n_snp, n_admix_haplo)

eur_hap = get_subset(eur_hap, n_snp, n_anc_haplo)
afr_hap = get_subset(afr_hap, n_snp, n_anc_haplo)

# make xarray
ds_admix = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "ploidy"), admix_hap),
        "lanc": (("indiv", "snp", "ploidy"), admix_lanc),
    },
)
ds_admix = ds_admix.assign_coords(
    {
        "snp": legend.id.values.astype(str),
        "indiv": ["admix_" + str(i) for i in np.arange(ds_admix.dims["indiv"])],
        "POS": ("snp", legend.position.values),
        "EUR_freq": ("snp", legend.EUR.values),
        "AFR_freq": ("snp", legend.AFR.values),
        "REF": ("snp", legend.a0.values.astype(str)),
        "ALT": ("snp", legend.a1.values.astype(str)),
    }
).assign_attrs(n_anc=2, description="Simulated African American individuals")

ds_eur = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "ploidy"), eur_hap),
    },
    coords={
        "snp": legend.id.values.astype(str),
        "indiv": ["eur_" + str(i) for i in np.arange(eur_hap.shape[0])],
        "POS": ("snp", legend.position.values),
        "REF": ("snp", legend.a0.values.astype(str)),
        "ALT": ("snp", legend.a1.values.astype(str)),
    },
    attrs={"n_anc": 1, "description": "Simulated EUR"},
)

ds_afr = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "ploidy"), afr_hap),
    },
    coords={
        "snp": legend.id.values.astype(str),
        "indiv": ["afr_" + str(i) for i in np.arange(afr_hap.shape[0])],
        "POS": ("snp", legend.position.values),
        "REF": ("snp", legend.a0.values.astype(str)),
        "ALT": ("snp", legend.a1.values.astype(str)),
    },
    attrs={"n_anc": 1, "description": "Simulated AFR"},
)


ds_admix.chunk(chunks={"snp": "auto"}).to_zarr("admix.zip", mode="w")
ds_eur.chunk(chunks={"snp": "auto"}).to_zarr("eur.zip", mode="w")
ds_afr.chunk(chunks={"snp": "auto"}).to_zarr("afr.zip", mode="w")
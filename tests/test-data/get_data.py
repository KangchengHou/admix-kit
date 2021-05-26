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
    return np.dstack((hap[:, 0:n_snp][np.arange(0, n_haplo, 2)],
                      hap[:, 0:n_snp][np.arange(1, n_haplo, 2)]))


admix_hap = get_subset(admix_hap, n_snp, n_admix_haplo)
admix_lanc = get_subset(admix_lanc, n_snp, n_admix_haplo)

eur_hap = get_subset(eur_hap, n_snp, n_anc_haplo)
afr_hap = get_subset(afr_hap, n_snp, n_anc_haplo)

# make xarray
ds_admix = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "haploid"), admix_hap),
        "lanc": (("indiv", "snp", "haploid"), admix_lanc)
    },
)
ds_admix = ds_admix.assign_coords({"snp": legend.id.values,
                                   "snp_position": ("snp", legend.position.values),
                                   "snp_eur_freq": ("snp", legend.EUR.values),
                                   "snp_afr_freq": ("snp", legend.AFR.values),
                                   "snp_a0": ("snp", legend.a0.values),
                                   "snp_a1": ("snp", legend.a1.values)}).assign_attrs(
    n_anc=2,
    description="Simulated African American individuals")

ds_eur = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "haploid"), eur_hap),
    },
    coords={"snp": legend.id.values,
            "position": ("snp", legend.position.values),
            "snp_a0": ("snp", legend.a0.values),
            "snp_a1": ("snp", legend.a1.values)},
    attrs={"n_anc": 1,
           "description": "Simulated EUR"}
)

ds_afr = xr.Dataset(
    data_vars={
        "geno": (("indiv", "snp", "haploid"), afr_hap),
    },
    coords={"snp": legend.id.values,
            "position": ("snp", legend.position.values),
            "snp_a0": ("snp", legend.a0.values),
            "snp_a1": ("snp", legend.a1.values)},
    attrs={"n_anc": 1,
           "description": "Simulated AFR"}
)

ds_admix.to_zarr("admix.zarr", mode='w')
ds_eur.to_zarr("eur.zarr", mode='w')
ds_afr.to_zarr("afr.zarr", mode='w')

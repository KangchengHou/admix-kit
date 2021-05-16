import pandas as pd
from pysnptools.snpreader import Bed
import numpy as np
from tqdm import tqdm


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


class Genotype(object):
    """
    Genotype object storing the genotype, snp information, individual information
    """

    def __init__(self, X=np.ndarray, indiv=None, snp=None) -> None:
        """

        Args:
            X
                A #indiv × #snp data matrix. A view of the data is used if the
                data type matches, otherwise, a copy is made.
            indiv
                Key-indexed one-dimensional observations annotation of length #indiv.
            snp
                Key-indexed one-dimensional variables annotation of length #snp.
        """
        self._X = X
        self._n_snp = len([] if snp is None else snp)
        self._n_indiv = len([] if indiv is None else indiv)
        shape = X.shape
        if shape is not None:
            if self._n_indiv == 0:
                self._n_indiv = shape[0]
            else:
                if self._n_indiv != shape[0]:
                    raise ValueError("`shape` is inconsistent with `indiv`")
            if self._n_snp == 0:
                self._n_snp = shape[1]
            else:
                if self._n_snp != shape[1]:
                    raise ValueError("`shape` is inconsistent with `snp`")

        # annotations
        self._indiv = _gen_dataframe(indiv, self._n_obs, ["obs_names", "row_names"])
        self._snp = _gen_dataframe(snp, self._n_vars, ["var_names", "col_names"])

        self.data = Bed(path)
        self.bim = pd.read_csv(
            path + ".bim",
            header=None,
            delim_whitespace=True,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        )
        read_plink()

    def __init__(
        self, path, indiv_index=None, snp_index=None, indiv_subset=None, snp_subset=None
    ):
        assert (indiv_index is None) or (indiv_subset is None)
        assert (snp_index is None) or (snp_subset is None)

        self.path = path
        self.data = Bed(path, count_A1=False)
        self.bim = pd.read_csv(
            path + ".bim",
            header=None,
            delim_whitespace=True,
            names=["CHR", "SNP", "CM", "BP", "A1", "A2"],
        )
        if indiv_subset is not None:
            indiv_index = self.get_index(indiv_subset=indiv_subset)

        if snp_subset is not None:
            snp_index = self.get_index(snp_subset=snp_subset)

        self.data = self.data[indiv_index, snp_index]
        if snp_index is not None:
            self.bim = self.bim.loc[snp_index, :].reset_index(drop=True)

        self.num_snps = self.data.col_count
        self.num_indivs = self.data.row_count

    def get_index(self, indiv_subset=None, snp_subset=None):
        """
        get index of the corresponding individual or SNPs
        """
        assert (indiv_subset is None) or (
            snp_subset is None
        ), "Feed only one of `indiv_subset` or `snp_subset`, if want both index, feed them sequentially"

        if indiv_subset is not None:
            all_indiv = [",".join(r) for r in self.data.iid.astype(str)]
            subset_indiv = [",".join(r) for r in indiv_subset.astype(str)]
            indiv_index = np.where(np.isin(all_indiv, subset_indiv))[0]
            return indiv_index

        if snp_subset is not None:
            snp_index = np.where(np.isin(self.data.sid.astype(str), snp_subset))[0]
            return snp_index

    def compute_mean_std(self, only_mean=False, chunk_size=500):
        """
        compute mean and standard deviation and store them
        """
        mean = np.zeros(self.num_snps)
        std = np.zeros(self.num_snps)

        for i in tqdm(range(0, self.num_snps, chunk_size)):
            sub_data = self.data[:, i : i + chunk_size].read().val
            sub_mean = np.nanmean(sub_data, axis=0)
            mean[i : i + chunk_size] = sub_mean
            if not only_mean:
                nanidx = np.where(np.isnan(sub_data))
                sub_data[nanidx] = sub_mean[nanidx[1]]
                std[i : i + chunk_size] = np.std(sub_data, axis=0)
        self.mean = mean
        if not only_mean:
            self.std = std

    def compute_corr(self, chunk_size=500):
        corr = np.zeros([self.num_snps, self.num_snps])
        for row_start in range(0, self.num_snps, chunk_size):
            for col_start in range(0, self.num_snps, chunk_size):
                # for each block
                row_stop = min(row_start + chunk_size, self.num_snps)
                col_stop = min(col_start + chunk_size, self.num_snps)

                row_geno = self.get(snp_index=np.arange(row_start, row_stop))
                col_geno = self.get(snp_index=np.arange(col_start, col_stop))

                corr[
                    np.ix_(
                        np.arange(row_start, row_stop), np.arange(col_start, col_stop)
                    )
                ] = (np.dot(row_geno.T, col_geno) / self.num_indivs)
        return corr

    def subset(self, indiv_index=None, snp_index=None):
        """
        return a Genotype object with the give indiv_index and snp_index
        they still share the same path
        """

        return Genotype(self.path, indiv_index=indiv_index, snp_index=snp_index)

    def get(self, indiv_index=None, snp_index=None, impute=True, std=True):
        """
        return a subset of genotype as numpy array
        """
        if indiv_index is None:
            indiv_index = np.arange(self.num_indivs)
        if snp_index is None:
            snp_index = np.arange(self.num_snps)
        sub_data = self.data[indiv_index, snp_index].read().val

        # impute
        if impute:
            nanidx = np.where(np.isnan(sub_data))
            sub_mean = self.mean[snp_index]
            sub_data[nanidx] = sub_mean[nanidx[1]]

        # standardize
        if std:
            sub_std = self.std[snp_index]
            sub_data = (sub_data - sub_mean) / sub_std
        return sub_data

    def to_plink(self, prefix):
        indiv_file = prefix + ".indiv"
        indiv_list = self.data.iid.astype(str)
        np.savetxt(indiv_file, indiv_list, fmt="%s")
        cmd = f"module load plink && plink --bfile {self.path} --make-bed --out {prefix} --keep {indiv_file} --keep-allele-order"
        return cmd

    # TODO:
    # def __getitem__(self, index: Index) -> "AnnData":
    #     """Returns a sliced view of the object."""
    #     oidx, vidx = self._normalize_indices(index)
    #     return AnnData(self, oidx=oidx, vidx=vidx, asview=True)

    # def _normalize_indices(self, index):
    #     pass
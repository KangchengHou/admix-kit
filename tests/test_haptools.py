"""
End-to-end test for simulating admixed genotypes with haptools.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

import admix


@pytest.mark.skipif(shutil.which("haptools") is None, reason="haptools not installed")
def test_haptools_simu_admix():
    """
    Simulated .lanc must match the simulated .pgen, and every local ancestry label
    must agree with the haptools .bp file (inclusive segment end positions).
    Regression test for https://github.com/KangchengHou/admix-kit/issues/43
    """
    pfile = os.path.join(admix.dataset.get_test_data_dir(), "toy-all")
    pos = pd.read_csv(pfile + ".pvar", sep="\t", comment="#", header=None)[1].values

    with tempfile.TemporaryDirectory() as tmp_dir:
        # synthetic genetic map (1 cM / Mb) in PLINK .map format
        mapdir = os.path.join(tmp_dir, "map")
        os.makedirs(mapdir)
        pd.DataFrame(
            {"chrom": 22, "id": ".", "cm": pos / 1e6, "pos": pos}
        ).to_csv(
            os.path.join(mapdir, "plink.chr22.GRCh37.map"),
            sep="\t",
            header=False,
            index=False,
        )
        out = os.path.join(tmp_dir, "sim")
        admix.tools.haptools_simu_admix(
            pfile=pfile,
            admix_prop={"CEU": 0.2, "YRI": 0.8},
            pop_col="Population",
            mapdir=mapdir,
            n_gen=8,
            n_indiv=20,
            out_prefix=out,
        )
        dset = admix.io.read_dataset(out, n_anc=2)
        assert dset.geno.shape == dset.lanc.shape == (len(pos), 20, 2)

        # expected labels from .bp: a SNP takes the ancestry of the first segment
        # whose (inclusive) end position is >= the SNP position
        anc_map = {"CEU": 0, "YRI": 1}
        haplos = []
        with open(out + ".bp") as f:
            for line in f:
                if line.startswith("Sample_"):
                    haplos.append([])
                else:
                    anc, _, end, _ = line.split()
                    haplos[-1].append((int(end), anc_map[anc]))
        expected = np.stack(
            [
                np.array([v for _, v in h])[
                    np.searchsorted([e for e, _ in h], pos, side="left")
                ]
                for h in haplos
            ],
            axis=1,
        )
        lanc = dset.lanc.compute().reshape(len(pos), -1)
        assert np.array_equal(lanc, expected)

# %load_ext autoreload
# %autoreload 2

from admix import data
import numpy as np
import pandas as pd
from os.path import join
import h5py
from admix.ancestry import WindowHMM, Lamp
import os
import admix
import json


def test_correct():
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(join(THIS_DIR, "data/test_correct.json")) as f:
        data = json.load(f)

    X = np.array(data["X"], dtype=np.int8)

    model = WindowHMM(
        data["model.n_snp"],
        data["model.n_proto"],
        emit_prior=0.0,
        trans_prior=0.0,
    )
    model._init_from_X(X)
    model._start = np.array(data["model.start_prob"])
    model._trans = np.array(data["model.trans_prob"])
    model._emit = np.array(data["model.emit_prob"])

    model.fit(X, max_iter=10, rel_tol=1e-6)

    assert np.allclose(model._start, np.array(data["fit.start_prob"]))
    assert np.allclose(model._trans, np.array(data["fit.trans_prob"]))
    assert np.allclose(model._emit, np.array(data["fit.emit_prob"]))


# ------
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# THIS_DIR = "/Users/kangchenghou/work/admix-tools/test/ancestry/"
# hap = admix.data.read_hap(join(THIS_DIR, "data/admix.hap"))
# lanc = admix.data.read_lanc(join(THIS_DIR, "data/admix.lanc"))
# afr_hap = admix.data.read_hap(join(THIS_DIR, "data/AFR.hap"))
# eur_hap = admix.data.read_hap(join(THIS_DIR, "data/EUR.hap"))

# rls = admix.ancestry.infer_local_ancestry(hap, [afr_hap, eur_hap])
# # some basic comparison between the `rls` and `lanc`


def speed():
    n_proto = 8
    n_snp = 300
    start = np.ones(n_proto) / n_proto
    A = np.random.random((n_proto, n_proto))
    A = A / A.sum(axis=1)[:, np.newaxis]
    trans = np.zeros((n_snp - 1, n_proto, n_proto))
    for snp_i in range(n_snp - 1):
        trans[snp_i, :, :] = A

    emit = np.random.random((n_snp, n_proto))
    model = WindowHMM(n_snp=n_snp, n_proto=n_proto)
    model._start = start
    model._trans = trans
    model._emit = emit
    n_indiv = 1000
    X, Z = model.sample(n_samples=n_indiv)

    model = WindowHMM(n_snp=n_snp, n_proto=n_proto, X=X)
    # %load_ext line_profiler
    # %lprun -f model.fit -f model._accum_suff_stats -f model._do_backward_pass model.fit(X=X, max_iter=10)
    # %lprun -f model.fit model.fit(X=X, max_iter=10)


def acc():
    data_dir = "/Users/kangchenghou/work/Lamp.jl/benchmark/test_data"
    ref_list = []
    for pop in ["EUR", "AFR"]:
        ref_list.append(admix.data.read_hap(join(data_dir, f"{pop}.ref")))
    snp_pos = np.loadtxt(join(data_dir, "snp.pos"), dtype=int)

    sample_hap = admix.data.read_hap(join(data_dir, "sample.EUR_0.5_AFR_0.5_50.hap"))
    sample_lanc = admix.data.read_hap(join(data_dir, "sample.EUR_0.5_AFR_0.5_50.anc"))
    model = Lamp(
        n_anc=len(ref_list),
        n_snp=len(snp_pos[0:600]),
        window_size=300,
        n_proto=8,
        snp_pos=snp_pos[0:600],
    )
    # %lprun -f model.fit -f WindowHMM.fit model.fit([ref for ref in ref_list])

    model.fit([ref[:, 0:600] for ref in ref_list])

    obs_prob, inferred_anc = model.predict(sample_hap[0:1, 0:600])

import numpy as np
from typing import List
from ._hmm import WindowHMM, decode_viterbi
from tqdm import tqdm


class Lamp(object):
    def __init__(
        self,
        n_anc: int,
        n_snp: int,
        snp_pos: np.ndarray,
        window_size: int = 300,
        n_proto: int = 6,
        smooth: bool = False,
        recomb_rate: float = 1e-8,
        verbose: bool = False,
        trans_prior: float = 0.00006,
        emit_prior: float = 0.001,
    ) -> None:
        self.n_anc = n_anc
        self.n_snp = n_snp
        self.n_proto = n_proto
        self.snp_pos = snp_pos
        self.recomb_rate = recomb_rate
        self.verbose = verbose
        self.n_window = int(self.n_snp / window_size)
        self.snp_index = np.array_split(np.arange(self.n_snp), self.n_window)
        self.trans_prior = trans_prior
        self.emit_prior = emit_prior

        self.smooth_snp_index = [
            np.arange(
                int(np.mean(self.snp_index[i])), int(np.mean(self.snp_index[i + 1]))
            )
            for i in range(self.n_window - 1)
        ]
        self.hmm_array = [
            [None for _ in range(self.n_anc)] for _ in range(self.n_window)
        ]
        self.smooth = smooth
        if self.smooth:
            self.smooth_hmm_array = [
                [None for _ in range(self.n_anc)] for _ in range(self.n_window)
            ]

    def fit(self, ref_list: List[np.ndarray]) -> None:
        for ref in ref_list:
            assert ref.shape[1] == self.n_snp
        assert len(ref_list) == self.n_anc

        for window_i in tqdm(range(self.n_window)):
            snp_index = self.snp_index[window_i]
            for anc_i in range(self.n_anc):
                ref_chunk = ref_list[anc_i][:, snp_index]
                hmm = WindowHMM(
                    n_snp=ref_chunk.shape[1],
                    n_proto=self.n_proto,
                    X=ref_chunk,
                    emit_prior=self.emit_prior,
                    trans_prior=self.trans_prior,
                )
                self.hmm_array[window_i][anc_i] = hmm.fit(X=ref_chunk)

        if self.smooth:
            for window_i in range(self.n_window - 1):
                snp_index = self.smooth_snp_index[window_i]
                for anc_i in range(self.n_anc):
                    ref_chunk = ref_list[anc_i][:, snp_index]
                    hmm = WindowHMM(
                        n_snp=ref_chunk.shape[1],
                        n_proto=self.n_proto,
                        X=ref_chunk,
                        emit_prior=self.emit_prior,
                        trans_prior=self.trans_prior,
                    ).fit()
                    self.smooth_hmm_array[window_i, anc_i] = hmm

    def predict(self, sample_hap: np.ndarray) -> None:
        assert sample_hap.shape[1] == self.n_snp
        n_sample = sample_hap.shape[0]
        obs_prob = np.zeros((n_sample, self.n_window, self.n_anc))
        trans_prob = np.zeros((self.n_window - 1, self.n_anc, self.n_anc))

        for window_i in range(self.n_window):
            snp_index = self.snp_index[window_i]
            # fill in the obs_prob
            for sample_i in range(n_sample):
                for anc_i in range(self.n_anc):
                    framelogprob = self.hmm_array[window_i][anc_i]._compute_log_lkl(
                        sample_hap[sample_i, snp_index]
                    )
                    fwdlattice, fwd_cond_prob = self.hmm_array[window_i][
                        anc_i
                    ]._do_forward_pass(framelogprob)
                    obs_prob[sample_i, window_i, anc_i] = sum(fwd_cond_prob)
            # fill in the transition prob
            if window_i > 0:
                snp_pos_gap = self.snp_pos[snp_index[0]] - self.snp_pos[snp_index[-1]]
                trans_base = self.recomb_rate * snp_pos_gap
                for anc_i in range(self.n_anc):
                    for anc_j in range(self.n_anc):
                        trans_prob[window_i - 1, anc_i, anc_j] = trans_base ** (
                            anc_i != anc_j
                        )

        inferred_anc = np.zeros((n_sample, self.n_snp), dtype=int)
        # return inferred_anc
        for sample_i in range(n_sample):
            decoded = decode_viterbi(
                np.zeros(self.n_anc), trans_prob, obs_prob[sample_i, :, :]
            )
            for window_i in range(self.n_window):
                inferred_anc[sample_i, self.snp_index[window_i]] = decoded[window_i]
        # TODO: incorporate smoothing procedure
        assert not self.smooth
        return obs_prob, inferred_anc

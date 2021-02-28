import numpy as np
from ._utils import normalize, log_mask_zero, log_normalize
from . import _hmmc


class WindowHMM:
    """HMM for Local Ancestry in Admixed Populations (LAMP)
    Parameters
    ----------
    n_proto: int
        Number of prototypical states
    trans_prior: float
        Prior / pseudocount for the transition probabilities between states.
    emission_prior: float
        Prior / pseudocount for the transition probabilities between states.
    random_state: int
        Seed
    max_iter: int, optional
        Maximum number of iterations to perform
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    init_params : string, optional
        Controls which parameters are initialized prior to
            training.  Can contain any combination of 's' for
            startprob, 't' for transmat, 'e' for emission. Default to None
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    start_prob_ : array, shape (n_proto, )
        Initial state occupation distribution.
    transition_prob_list_ : n_snp-length List of array, each array (n_proto, n_proto)
        Matrix of transition probabilities between states.
    emission_prob_list_ : array, shape (n_proto, )
        Probability of emitting a given symbol when in each state.
    Examples
    --------
    TODO
    """

    def __init__(
        self,
        n_snp: int,
        n_proto: int,
        X: np.ndarray = None,
        trans_prior: float = 0.1,
        emit_prior: float = 0.1,
        founder_bias: float = 0.9,
    ):
        """Initialize

        Args:
            X (np.ndarray): input dataset
            n_proto (int): Number of prototypical states
            trans_prior (float, optional): transition prior. Defaults to 0.1.
            emit_prior (float, optional): emission prior. Defaults to 0.1.
            founder_bias (float, optional): founder bias. Defaults to 0.9.
        """

        self.n_snp = n_snp
        self.n_proto = n_proto
        self.trans_prior = trans_prior
        self.emit_prior = emit_prior
        self.founder_bias = founder_bias

        if X is not None:
            # initialize with the original haplotype matrix
            assert self.n_snp == X.shape[1]
            self._init_from_X(X)

    def _init_from_X(self, X: np.ndarray):
        """Initilize with reference data

        Args:
            X (np.ndarray): [input dataset]
        """
        start = np.exp(np.random.uniform(low=-1.0, high=1.0, size=self.n_proto))
        start /= np.sum(start)
        trans = np.zeros((self.n_snp - 1, self.n_proto, self.n_proto))
        for i in range(self.n_snp - 1):
            tmp = (
                np.ones((self.n_proto, self.n_proto))
                * (1.0 - self.founder_bias)
                / (self.n_proto - 1)
            )
            tmp[np.diag_indices(self.n_proto)] = self.founder_bias
            tmp *= np.exp(
                np.random.uniform(low=-1.0, high=1.0, size=(self.n_proto, self.n_proto))
            )
            trans[i, :, :] = normalize(tmp, axis=1)

        tmp = (np.sum(X, axis=0) + 1) / (X.shape[0] + 2)
        emit = np.zeros((self.n_snp, self.n_proto))
        for i in range(self.n_proto):
            tmp1 = tmp * np.exp(np.random.uniform(-1, 1, size=self.n_snp))
            tmp2 = (1 - tmp) * np.exp(np.random.uniform(-1, 1, size=self.n_snp))
            emit[:, i] = tmp1 / (tmp1 + tmp2)

        self._start = start
        self._trans = trans
        self._emit = emit

    def fit(self, X: np.ndarray, max_iter: int = 20, rel_tol: float = 0.01):
        """Estimate model parameters.
        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.
        Parameters
        ----------
        X : array-like, shape (n_indiv, n_snp)
            Feature matrix of individual samples.
        Returns
        -------
        self : object
            Returns self.
        """

        assert self.n_snp == X.shape[1]
        n_indiv = X.shape[0]

        prev_total_logprob = np.finfo("d").min

        for iter_i in range(max_iter):
            self._init_suff_stats()
            # E-step

            ########### compressed code
            # total_logprob = _hmmc._fit_iter(
            #     self.n_snp,
            #     self.n_proto,
            #     X.shape[0],
            #     X,
            #     self._start,
            #     self._trans,
            #     self._emit,
            #     self._suff_stats["start"],
            #     self._suff_stats["trans"],
            #     self._suff_stats["emit"],
            # )

            ############# PREVIOUS CODE
            total_logprob = 0.0
            for indiv_i in range(n_indiv):
                log_obs = self._compute_log_lkl(X[indiv_i, :])
                fwd_lattice, fwd_cond_prob = self._do_forward_pass(log_obs)
                total_logprob += sum(fwd_cond_prob)
                bwd_lattice, bwd_cond_prob = self._do_backward_pass(log_obs)
                posteriors = self._compute_posteriors(fwd_lattice, bwd_lattice)

                self._accum_suff_stats(
                    X=X[indiv_i, :],
                    log_obs=log_obs,
                    posteriors=posteriors,
                    fwd_lattice=fwd_lattice,
                    bwd_lattice=bwd_lattice,
                )

            self._do_mstep()
            if (prev_total_logprob - total_logprob) / prev_total_logprob < rel_tol:
                break

        return self

    def _do_forward_pass(self, log_obs: np.ndarray):
        """Forward pass

        Args:
            log_obs (np.ndarray): [n_snp x n_proto observation likelihood]

        Returns:
            [type]: [description]
        """
        assert log_obs.shape == (self.n_snp, self.n_proto)
        fwd_lattice = np.zeros((self.n_snp, self.n_proto))
        fwd_cond_prob = np.zeros(self.n_snp)
        _hmmc._forward(
            self.n_snp,
            self.n_proto,
            self._start,
            self._trans,
            log_obs,
            fwd_lattice,
            fwd_cond_prob,
        )
        return fwd_lattice, fwd_cond_prob

    def _do_backward_pass(self, log_obs):

        assert log_obs.shape == (self.n_snp, self.n_proto)

        bwd_lattice = np.zeros((self.n_snp, self.n_proto))
        bwd_cond_prob = np.zeros(self.n_snp)
        L = np.zeros(self.n_proto)

        _hmmc._backward(
            self.n_snp,
            self.n_proto,
            self._trans,
            log_obs,
            bwd_lattice,
            bwd_cond_prob,
            L,
        )

        return bwd_lattice, bwd_cond_prob

    def _init_suff_stats(self):
        self._suff_stats = {
            "start": np.zeros(self.n_proto),
            "trans": np.zeros((self.n_snp - 1, self.n_proto, self.n_proto)),
            "emit": np.zeros((self.n_snp, self.n_proto, 2)),
        }

    def _accum_suff_stats(self, X, log_obs, posteriors, fwd_lattice, bwd_lattice):
        # update start
        self._suff_stats["start"] += posteriors[0, :]

        # update transition
        xi = np.zeros_like(self._trans)
        L = np.zeros(self.n_proto)
        _hmmc._compute_xi(
            self.n_snp,
            self.n_proto,
            fwd_lattice,
            self._trans,
            bwd_lattice,
            log_obs,
            xi,
            L,
        )
        self._suff_stats["trans"] += xi

        # update emission
        _hmmc._update_emit(
            self.n_snp, self.n_proto, X, self._suff_stats["emit"], posteriors
        )

    def _do_mstep(self):

        # TODO: add prior
        self._start = self._suff_stats["start"] / sum(self._suff_stats["start"])

        self._trans = np.zeros((self.n_snp - 1, self.n_proto, self.n_proto))
        # TODO: check the correctness
        for i in range(self.n_snp - 1):
            self._trans[i, :, :] = normalize(self._suff_stats["trans"][i, :, :], axis=1)

        self._emit = self._suff_stats["emit"][:, :, 1] / (
            self._suff_stats["emit"][:, :, 0] + self._suff_stats["emit"][:, :, 1]
        )

    def sample(self, n_samples=1, seed=1234):
        """Generate random samples from the model.
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        Returns
        -------
        X : array, shape (n_samples, n_snp)
            Feature matrix.
        Z :
        """
        # NOTE: this could be speedup, check https://github.com/hmmlearn/hmmlearn/blob/da0fdf7f04c8aab4e7404dec586a27cdb371a7f2/hmmlearn/base.py#L217
        # where one can precompute the CDF and sample from that.
        np.random.seed(seed)

        all_X = []
        all_Z = []

        for _ in range(n_samples):
            X = []
            Z = []
            for snp_i in range(self.n_snp):
                if snp_i == 0:
                    Z.append(np.random.choice(self.n_proto, size=1, p=self._start)[0])
                else:
                    Z.append(
                        np.random.choice(
                            self.n_proto,
                            size=1,
                            p=self._trans[snp_i - 1, Z[-1], :],
                        )[0]
                    )
                X.append(np.random.binomial(n=1, p=self._emit[snp_i][Z[-1]]))

            all_X.append(X)
            all_Z.append(Z)
        return np.vstack(all_X).astype(np.int8), np.vstack(all_Z).astype(np.int8)

    def score(self, X):
        """Compute the log probability under the model.
        Parameters
        ----------
        X : SNP information, same length as `self.n_snp`
        Returns
        -------
        logprob : float
            Log likelihood of ``X``.
        """
        # calculate the observation probability
        log_obs = self._compute_log_likelihood(X)
        logprob, _fwd_lattice = self._do_forward_pass(log_obs)
        return logprob

    def decode(self, X):
        """Find most likely state sequence corresponding to ``X`` using Viterbi algorithm.
        Parameters
        ----------
        X : array-like, shape (n_snps, )
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.
        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.
        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        assert len(X) == self.n_snp
        state_sequence = np.empty(self.n_snp, dtype=np.int32)
        viterbi_lattice = np.zeros((self.n_snp, self.n_proto))
        work_buffer = np.empty(self.n_proto)
        log_obs = self._compute_log_likelihood(X)
        log_startprob = log_mask_zero(self.start_prob_)
        log_transition_prob_list = [
            log_mask_zero(tp) for tp in self.transition_prob_list_
        ]
        for i in range(self.n_proto):
            viterbi_lattice[0, i] = log_startprob[i] + log_obs[0, i]

        # Induction
        for t in range(1, self.n_snp):
            # for each i in z_{s+1}
            for i in range(self.n_proto):
                for j in range(self.n_proto):
                    work_buffer[j] = (
                        log_transition_prob_list[t - 1][j, i]
                        + viterbi_lattice[t - 1, j]
                    )

                viterbi_lattice[t, i] = max(work_buffer) + log_obs[t, i]

        # Observation traceback
        state_sequence[self.n_snp - 1] = where_from = np.argmax(
            viterbi_lattice[self.n_snp - 1]
        )
        logprob = viterbi_lattice[self.n_snp - 1, where_from]

        for t in range(self.n_snp - 2, -1, -1):
            for i in range(self.n_proto):
                work_buffer[i] = (
                    viterbi_lattice[t, i] + log_transition_prob_list[t][i, where_from]
                )
                work_buffer[i] = viterbi_lattice[t, i]
            state_sequence[t] = where_from = np.argmax(work_buffer)

        return logprob, state_sequence

    # def _init_fit(self, random=True):
    #     if "s" in self.init_param:
    #         if random:
    #             self.start_prob_ = normalize(np.random.rand(self.n_proto))
    #         else:
    #             self.start_prob_ = np.full(self.n_proto, 1 / self.n_proto)
    #     if "t" in self.init_param:
    #         if random:
    #             self.transition_prob_list_ = [
    #                 normalize(np.random.rand(self.n_proto, self.n_proto), axis=1)
    #                 for snp_i in range(self.n_snp - 1)
    #             ]
    #         else:
    #             self.transition_prob_list_ = [
    #                 np.full((self.n_proto, self.n_proto), 1 / self.n_proto)
    #                 for snp_i in range(self.n_snp - 1)
    #             ]
    #     if "e" in self.init_param:
    #         self.emission_prob_list_ = [
    #             np.random.rand(self.n_proto) for snp_i in range(self.n_snp)
    #         ]

    def _compute_posteriors(
        self, fwd_lattice: np.ndarray, bwd_lattice: np.ndarray
    ) -> np.ndarray:
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        gamma = fwd_lattice * bwd_lattice
        gamma = normalize(gamma, axis=1)
        return gamma

    def _compute_log_lkl(self, X: np.ndarray) -> np.ndarray:
        """Computes per-component log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_snp, )
            Feature matrix of individual samples.
        Returns
        -------
        logprob : array, shape (n_snp, n_proto)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        logprob = np.zeros((self.n_snp, self.n_proto))
        _hmmc._compute_log_lkl(self.n_snp, self.n_proto, X, self._emit, logprob)
        return logprob
        assert len(X) == self.n_snp

        logprob = []
        for snp_i in range(self.n_snp):
            if X[snp_i] == 1:
                emit_prob = self._emit[snp_i]
            else:
                emit_prob = 1 - self._emit[snp_i]
            logprob.append(log_mask_zero(emit_prob))
        return np.vstack(logprob)

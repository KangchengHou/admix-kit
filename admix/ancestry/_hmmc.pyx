# cython: language_level=3, boundscheck=False, wraparound=False
from cython cimport view
from cython.parallel import prange
from numpy.math cimport expl, logl, log1pl, isinf, fabsl, INFINITY
import numpy as np
cimport numpy as np
import time

ctypedef double dtype_t

cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]


# cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
#     cdef dtype_t X_max = _max(X)
#     if isinf(X_max):
#         return -INFINITY

#     cdef dtype_t acc = 0
#     for i in range(X.shape[0]):
#         acc += expl(X[i] - X_max)

#     return logl(acc) + X_max

# cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
#     if isinf(a) and a < 0:
#         return b
#     elif isinf(b) and b < 0:
#         return a
#     else:
#         return max(a, b) + log1pl(expl(-fabsl(a - b)))

"""
Fitting procedure for one iteration
Args:
    X: (n_indiv, n_snp) array
    start: (n_proto, ) array
    trans: (n_snp - 1, n_proto, n_proto) array
    emit: (n_snp, n_proto) array
Returns (update inplace):
    start: (n_proto, ) array
    trans: (n_snp - 1, n_proto, n_proto) array
    emit: (n_snp, n_proto) array
"""
# def _fit_iter(
#     int n_snp, 
#     int n_proto, 
#     int n_indiv,
#     np.int8_t[:, :] X,
#     dtype_t[:] start,
#     dtype_t[:, :, :] trans,
#     dtype_t[:, :] emit,
#     dtype_t[:] stats_start,
#     dtype_t[:, :, :] stats_trans,
#     dtype_t[:, :, :] stats_emit,
# ):
#     cdef int indiv_i, t, i, j
#     cdef dtype_t total_logprob

#     # TODO: what's ::view.contiguous?
#     cdef dtype_t[:, ::view.contiguous] log_obs = np.zeros((n_snp, n_proto))
#     cdef dtype_t[:, ::view.contiguous] fwd_lattice = np.zeros((n_snp, n_proto))
#     cdef dtype_t[:, ::view.contiguous] bwd_lattice = np.zeros((n_snp, n_proto))
#     cdef dtype_t[:, ::view.contiguous] posteriors = np.zeros((n_snp, n_proto))
#     cdef dtype_t[::view.contiguous] fwd_cond_prob = np.zeros(n_snp)
#     cdef dtype_t[::view.contiguous] bwd_cond_prob = np.zeros(n_snp)
#     cdef dtype_t[::view.contiguous] L = np.zeros(n_proto)
#     cdef dtype_t[:, :, ::view.contiguous] xi = np.zeros((n_snp - 1, n_proto, n_proto))


#     # E-step
#     total_logprob = 0.0
#     for indiv_i in range(n_indiv):
#         # calculate log likelihood of the observation
#         _compute_log_lkl(n_snp, n_proto, X[indiv_i, :], emit, log_obs)
#         # forward pass
#         _forward(n_snp, n_proto, start, trans, log_obs, fwd_lattice, fwd_cond_prob)
#         # calculate total logprob
#         for t in range(n_snp):
#             total_logprob += fwd_cond_prob[t]
#         # backward pass
#         _backward(n_snp, n_proto, trans, log_obs, bwd_lattice, bwd_cond_prob, L)
#         # compute posteriors
#         _compute_posteriors(n_snp, n_proto, fwd_lattice, bwd_lattice, posteriors)
#         # update start
#         for i in range(n_proto):
#             stats_start[i] += posteriors[0, i]
#         # update transition
#         _compute_xi(
#             n_snp,
#             n_proto,
#             fwd_lattice,
#             trans,
#             bwd_lattice,
#             log_obs,
#             xi,
#             L,
#         )
        
#         for t in range(n_snp - 1):
#             for i in range(n_proto):
#                 for j in range(n_proto):
#                     stats_trans[t][i][j] += xi[t][i][j]
#         # update emission
#         for t in range(n_snp):
#             for i in range(n_proto):
#                 stats_emit[t, i, X[indiv_i, t]] += posteriors[t, i]
#     return total_logprob

def _compute_posteriors(
    int n_snp,
    int n_proto,
    dtype_t[:, :] fwd_lattice,
    dtype_t[:, :] bwd_lattice,
    dtype_t[:, :] posteriors,
):
    cdef int i, j
    cdef dtype_t s
    for i in range(n_snp):
        s = 0
        # fwd_lattice * bwd_lattice
        for j in range(n_proto):
            posteriors[i][j] = fwd_lattice[i][j] * bwd_lattice[i][j]
            s += posteriors[i][j]
        # normalize
        for j in range(n_proto):
            posteriors[i][j] /= s
        
def _forward(int n_snp, int n_proto,
             dtype_t[:] start,
             dtype_t[:, :, :] trans,
             dtype_t[:, :] log_obs,
             dtype_t[:, :] fwdlattice,
             dtype_t[:] c):
    """
    n_snp: length of sequence
    n_proto: number of states
    start: `(n_proto, )` length
    trans: `(n_snp - 1, n_proto, n_proto)` matrix
    log_obs: `(n_snp, n_proto)` observation likelihood matrix
    fwdlattice: return matrix `(n_snp, n_proto)`
    c: normalization constant
    """

    cdef int t, i, j
    cdef dtype_t m
    c[:] = 0.0
    fwdlattice[:, :] = 0.0
    m = _max(log_obs[0, :])
    
    # Initialization
    for j in range(n_proto):
        fwdlattice[0, j] = start[j] * expl(log_obs[0, j] - m)
        c[0] += fwdlattice[0, j]
    
    for j in range(n_proto):
        fwdlattice[0, j] /= c[0]
    
    c[0] = logl(c[0]) + m
    
    # Induction
    for t in range(n_snp - 1):
        m = _max(log_obs[t + 1, :])
        for j in range(n_proto):
            for i in range(n_proto):
                fwdlattice[t + 1, j] += fwdlattice[t, i] * trans[t, i, j]
            fwdlattice[t + 1, j] *= expl(log_obs[t + 1, j]  - m)
            c[t + 1] += fwdlattice[t + 1, j]
        
        for j in range(n_proto):
            fwdlattice[t + 1, j] /= c[t + 1]
        
        c[t + 1] = logl(c[t + 1]) + m
    
    
    # TODO: what's dtype_t[::view.contiguous]


def _backward(int n_snp, int n_proto,
              dtype_t[:, :, :] trans,
              dtype_t[:, :] log_obs,
              dtype_t[:, :] bwdlattice,
              dtype_t[:] c,
              dtype_t[:] L):
    """
    n_snp: length of sequence
    n_proto: number of states
    trans: `(n_snp - 1, n_proto, n_proto)` matrix
    log_obs: `(n_snp, n_proto)` observation likelihood matrix
    bwdlattice: return matrix `(n_snp, n_proto)`
    """
    cdef int t, i, j
    cdef dtype_t m
    c[:] = 0.0
    bwdlattice[:, :] = 0.0
    # initilize
    for j in range(n_proto):
        bwdlattice[n_snp - 1, j] = 1.0

    # TODO: check start and end
    for t in range(n_snp - 2, -1, -1):
        m = _max(log_obs[t + 1, :])
        for i in range(n_proto):
            L[i] = expl(log_obs[t+1, i] - m)

        for j in range(n_proto):
            for i in range(n_proto):
                bwdlattice[t, j] += bwdlattice[t+1, i] * trans[t, j, i] * L[i]
            c[t+1] += bwdlattice[t, j]

        for j in range(n_proto):
            bwdlattice[t, j] /= c[t+1]

        c[t+1] = logl(c[t+1]) + m


def _compute_xi(int n_snp, int n_proto,
                dtype_t[:, :] fwdlattice,
                dtype_t[:, :, :] trans,
                dtype_t[:, :] bwdlattice,
                dtype_t[:, :] log_obs,
                dtype_t[:, :, :] xi,
                dtype_t[:] L
    ):

    cdef int t, i, j
    cdef dtype_t c, m
    for t in range(n_snp - 1):
        c = 0.
        m = _max(log_obs[t + 1, :])
        
        for j in range(n_proto):
            L[j] = expl(log_obs[t + 1, j] - m)
        
        for j in range(n_proto):
            for i in range(n_proto):
                xi[t, i, j] = fwdlattice[t, i] * trans[t, i, j] * L[j] * bwdlattice[t + 1, j]
                c += xi[t, i, j]
        
        for i in range(n_proto):
            for j in range(n_proto):
                xi[t, i, j] /= c

def _update_emit(int n_snp, int n_proto, np.int8_t[:] X, dtype_t[:, :, :] emit, dtype_t[:, :] posteriors):
    cdef int t, i
    for t in range(n_snp):
        for i in range(n_proto):
            emit[t, i, X[t]] = emit[t, i, X[t]] + posteriors[t, i]
    
def _compute_log_lkl(int n_snp, int n_proto, np.int8_t[:] X, dtype_t[:, :] emit, dtype_t[:, :] log_prob):
    """
    X: (n_snp, ) vector
    emit: (n_snp, n_proto)
    log_prob: (n_snp, n_proto)
    """
    cdef int i, j
    for i in range(n_snp):
        if X[i] == 1:
            for j in range(n_proto):
                log_prob[i, j] = logl(emit[i, j])
        else:
            for j in range(n_proto):
                log_prob[i, j] = logl(1 - emit[i, j])
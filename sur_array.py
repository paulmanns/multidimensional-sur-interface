import numpy as np
import copy
from enum import Enum

import sur_array_python
import hilbert_python


class Serialization(Enum):
    ROWWISE = 0
    COLUMNWISE = 1
    ALONG_HILBERT = 2
    RANDOM_PERM = 3


class SurArray:
    def __init__(self):
        pass

    # takes a list of matrices in shape (2^n, 2^n) and reduces it to a list of matrices in shape (2^k, 2^k) using the average
    def compute_fun_avg(self, lo_mat, n, k):
        for mat in lo_mat:
            assert mat.shape == (2**n, 2**n)
        assert k <= n
        new_lo_mat = []
        for mat in lo_mat:
            new_mat = np.zeros((2**k, 2**k))
            for i in range(2**k):
                for j in range(2**k):
                    new_mat[i, j] = np.mean(mat[i*2**(n-k):(i+1)*2**(n-k), j*2**(n-k):(j+1)*2**(n-k)])
            new_lo_mat.append(new_mat)
        return new_lo_mat

    # serializes a list of matrices in shape (2^k, 2^k) into a 1-dimensional array following a specific way through the matrices
    def serialize(self, lo_mat, k, variant):
        for mat in lo_mat:
            assert mat.shape == (2**k, 2**k)
        assert variant in {0, 1, 2, 3}
        lo_vec = []
        if variant == Serialization.ROWWISE.value:
            for mat in lo_mat:
                lo_vec.append(np.reshape(mat, 2**k * 2**k, order='C'))
            return lo_vec, None
        elif variant == Serialization.COLUMNWISE.value:
            for mat in lo_mat:
                lo_vec.append(np.reshape(mat, 2**k * 2**k, order='F'))
            return lo_vec, None
        elif variant == Serialization.ALONG_HILBERT.value:
            ctrs = np.zeros((2**k * 2**k, 2))
            ctrs = hilbert_python.coordinates_from_distance(ctrs, 2, k)
            for mat in lo_mat:
                new_vec = np.zeros(2**k * 2**k)
                for i in range(len(ctrs)):
                    new_vec[i] = mat[int(ctrs[i][0]), int(ctrs[i][1])]
                lo_vec.append(new_vec)
            return lo_vec, None
        elif variant == Serialization.RANDOM_PERM.value:
            lo_perm_inv = []
            for mat in lo_mat:
                temp_vec = np.reshape(mat, 2**k * 2**k, order='C')
                ind_perm = np.random.permutation(np.arange(2**k * 2**k))
                lo_perm_inv.append(np.argsort(ind_perm))
                lo_vec.append(temp_vec[ind_perm])
            return lo_vec, lo_perm_inv

    # computes SUR with alpha being a 2-dimensional array rowwise all serialized matrices of the lists before
    # if use_cython is set True, a cython module will be used
    # returns three 2-dimensional arrays (omega, gamma, phi) in the shape of alpha, again rowwise the information according to one matrix
    def compute_sur_omega(self, alpha, use_cython):
        n_elem = alpha.shape[1]
        omega = np.zeros(alpha.shape)
        phi = np.zeros(alpha.shape)
        gamma = copy.deepcopy(alpha)
        if use_cython:
            sur_array_python.compute_sur_omega(alpha.shape[0], alpha.shape[1], omega, phi, gamma)
        else:
            for i in range(0, n_elem):
                # ------------------------------------------------------------------ #
                # numpy.argmax, Note:                                                #
                # In case of multiple occurrences of the maximum values, the indices #
                # corresponding to the first occurrence are returned.                #
                # ------------------------------------------------------------------ #
                omega[np.argmax(gamma[:, i]), i] = 1.
                phi[:, i] = gamma[:, i] - omega[:, i]
                if i < n_elem - 1:
                    gamma[:, i + 1] += phi[:, i]
        return omega, gamma, phi

    # deserializes a list of 1-dimensional arrays into a list of the original matrices in shape (2^k, 2^k) following a specific way through the matrices
    def deserialize(self, lo_vec, k, variant, lo_perm_inv=None):
        for vec in lo_vec:
            assert len(vec) == 2**k * 2**k
        assert variant in {0, 1, 2, 3}
        lo_old_mat = []
        if variant == Serialization.ROWWISE.value:
            for vec in lo_vec:
                lo_old_mat.append(np.reshape(vec, (2**k, 2**k), order='C'))
        elif variant == Serialization.COLUMNWISE.value:
            for vec in lo_vec:
                lo_old_mat.append(np.reshape(vec, (2**k, 2**k), order='F'))
        elif variant == Serialization.ALONG_HILBERT.value:
            ctrs = np.zeros((2**k * 2**k, 2))
            ctrs = hilbert_python.coordinates_from_distance(ctrs, 2, k)
            for vec in lo_vec:
                old_mat = np.zeros((2**k, 2**k))
                for i in range(len(ctrs)):
                    old_mat[int(ctrs[i][0]), int(ctrs[i][1])] = vec[i]
                lo_old_mat.append(old_mat)
        elif variant == Serialization.RANDOM_PERM.value:
            for vec in lo_vec:
                assert lo_perm_inv is not None
                for perm_inv in lo_perm_inv:
                    temp_vec = vec[perm_inv]
                    lo_old_mat.append(np.reshape(temp_vec, (2**k, 2**k), order='C'))
        return lo_old_mat

    # takes a list of matrices in shape (2^k, 2^k) and returns a list of matrices in shape (2^n, 2^n) putting the average into all according matrix elements
    def scatter_to_fun(self, lo_mat, n, k):
        for mat in lo_mat:
            assert mat.shape == (2**k, 2**k)
        lo_new_mat = []
        for mat in lo_mat:
            new_mat = np.zeros((2**n, 2**n))
            for i in range(2**k):
                for j in range(2**k):
                    new_mat[i*2**(n-k):(i+1)*2**(n-k), j*2**(n-k):(j+1)*2**(n-k)].fill(mat[i, j])
            lo_new_mat.append(new_mat)
        return lo_new_mat

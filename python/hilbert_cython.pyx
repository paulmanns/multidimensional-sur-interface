from __future__ import print_function
cimport hilbert as hilbert_wrapper

import numpy as np
cimport numpy as np

def coordinates_from_distance(np.ndarray[double, ndim=2, mode="c"] input not None, int n, int p):
    hilbert_wrapper.c_coordinates_from_distance(&input[0,0],n,p)
    return input

def compute_ctrs(np.ndarray[double, ndim=2, mode="c"] ctrs not None, np.ndarray[double, ndim=1, mode="c"] widthArray not None, int hc_iter):
    width = 2. ** (-hc_iter)
    widthArray[hc_iter-1] = width
    #n_ctr = 2 ** (hc_iter * dim)
    ctrs = coordinates_from_distance(ctrs, 2, hc_iter)
    ctrs *= width
    ctrs += .5 * width
    return ctrs, widthArray

def compute_idx_table(double width, np.ndarray[int, ndim=1, mode="c"] ind_old not None, np.ndarray[int, ndim=2, mode="c"] indices not None, np.ndarray[int, ndim=1, mode="c"] indices_inv not None, int hc_iter, np.ndarray[double, ndim=2, mode="c"] coor not None, np.ndarray[double, ndim=2, mode="c"] ctrs not None, int dim, int n_ctr, int coor_dim):
    assert hc_iter != 1
    hilbert_wrapper.c_compute_idx_table(width, &ind_old[0], &indices[0,0], &indices_inv[0], &coor[0,0], &ctrs[0,0], hc_iter, dim, n_ctr, coor_dim)
    return indices, indices_inv

def compute_idx_table_slow(double width, np.ndarray[int, ndim=2, mode="c"] indices not None, np.ndarray[int, ndim=1, mode="c"] indices_inv not None, int hc_iter, np.ndarray[double, ndim=2, mode="c"] coor not None, np.ndarray[double, ndim=2, mode="c"] ctrs not None, int dim, int n_ctr, int coor_dim):
    assert hc_iter == 1
    hilbert_wrapper.c_compute_idx_table_slow(width, &indices[0,0], &indices_inv[0], &coor[0,0], &ctrs[0,0], hc_iter, dim, n_ctr, coor_dim)
    return indices, indices_inv

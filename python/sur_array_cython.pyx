from __future__ import print_function
cimport sur_array as sur_array_wrapper

import numpy as np
cimport numpy as np


def compute_sur_omega(int rows, int cols, np.ndarray[double, ndim=2, mode="c"] omega not None, np.ndarray[double, ndim=2, mode="c"] phi not None, np.ndarray[double, ndim=2, mode="c"] gamma not None, int vc=0):
    sur_array_wrapper.c_compute_sur_omega(rows, cols, &omega[0,0], &phi[0,0], &gamma[0,0], vc)
    return omega, phi, gamma

def compute_median_filter(int hc_iter, np.ndarray[double, ndim=1, mode="c"] vec not None, np.ndarray[double, ndim=2, mode="c"] ctrs not None, int lenCtrs, np.ndarray[double, ndim=1, mode="c"] avg not None, double width, np.ndarray[int, ndim=2, mode="c"] idx_table not None, int len1IdxTable, int len2IdxTable):
    sur_array_wrapper.c_compute_median_filter(hc_iter, &vec[0], &ctrs[0,0], lenCtrs, &avg[0], width, &idx_table[0,0], len1IdxTable, len2IdxTable)

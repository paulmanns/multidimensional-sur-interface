cdef extern from "sur_array.h":
  void c_compute_sur_omega(int rows, int cols, double* omega, double* phi, double* gamma, int vc)
  void c_compute_median_filter(int hc_iter, double* vec, double* ctrs, int lenCtrs, double* avg, double width, int* idx_table, int len1IdxTable, int len2IdxTable)



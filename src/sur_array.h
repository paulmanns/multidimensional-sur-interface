#ifndef SUR_ARRAY
#define SUR_ARRAY_h

void c_compute_sur_omega(int rows, int cols, double* omega, double* phi, double* gamma, int vc);
void c_compute_median_filter(int hc_iter, double* vec, double* ctrs, int len_ctrs, double* avg, double width, int* idx_table, int len1IdxTable, int len2IdxTable);

#endif // SUR_ARRAY_h


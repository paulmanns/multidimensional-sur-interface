#ifndef HILBERT_h
#define HILBERT_h

void c_coordinates_from_distance(double* pts, int n, int p);
void c_compute_idx_table(double width, int* indices_old, int* indices, int* indices_inv, double* coor, double* ctrs, int hc_iter, int dim, int n_ctr, int coor_dim);
void c_compute_idx_table_slow(double width, int* indices, int* indices_inv, double* coor, double* ctrs, int hc_iter, int dim, int n_ctr, int coor_dim);

#endif // HILBERT_h


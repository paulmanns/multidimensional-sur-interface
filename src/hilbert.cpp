#include "hilbert.h"
#include <iostream>
#include <limits>

#include <eigen3/Eigen/Dense>

void c_coordinates_from_distance(double* pts, int n, int p){

	unsigned int L = n*p;  //Length of the binary representations

	for ( int h = 0; h < (1<<(n*p)); h++ )
	{
		int x[n]; //coordinates of h'th point along Hilbert Curve
		//all n coordinates
		for ( int i = 0; i < n; i++ ){
			//determine the mask 
			int mask = 0;
			for ( int j = 0; j < L/n; j++ ) 
				mask |= ( 1 << (L-1)-i-j*n );

			//determine the i'th coordinate of x
			int result = 0;
			int v = mask & h;	
			v >>= n-(i+1);
			for ( int k = 0; k <= L-1; k++ ){
				result += (v & 1) * (1 << k);
				v >>= n;
			}
			x[i] = result;
		}

		int t = x[n-1] >> 1;
		for ( int i = n-1; i > 0; i-- ) {
			x[i] ^= x[i-1];
		}
		x[0] ^= t;		

		int Z = 2 << (p-1);
		int Q = 2;
		while( Q != Z ){
			int P = Q-1;
			for ( int i = n-1; i >= 0; i-- ) {
				if ( x[i] & Q ) {
					//invert
		            x[0] ^= P;
				} else {
					//exchange
					t = (x[0] ^ x[i]) & P;
					x[0] ^= t;
					x[i] ^= t;
				}
			}
			Q <<= 1;
		}
		
		for ( int i = 0; i < n; i++ ) {
			pts[h*n+i] = x[i]; 
		}
	}
    return;
}

void c_compute_idx_table(double width, int* indices_old, int* indices, int* indices_inv, double* coor, double* ctrs, int hc_iter, int dim, int n_ctr, int coor_dim)
{
    Eigen::VectorXd c = Eigen::ArrayXd::Zero(n_ctr);
    Eigen::VectorXd v(2);

    for (int i = 0; i < dim; i++) {
        int j_prev = int(indices_old[i]);
        for (int j = 4*j_prev; j < 4*(j_prev+1); j++) {
            v << coor[2*i]-ctrs[2*j], coor[2*i+1]-ctrs[2*j+1];
            if ( v.lpNorm<Eigen::Infinity>() <= 0.5 * width + Eigen::NumTraits<float>::epsilon() ) {
                indices[j*(coor_dim / n_ctr) + int(c[j])] = i;
                indices_inv[i] = j;
                c[j] += 1;
            }

        }
    }
    for (int i = 0; i < n_ctr; i++) {
        assert(c[i] == coor_dim / n_ctr);
    }

    return;
}

void c_compute_idx_table_slow(double width, int* indices, int* indices_inv, double* coor, double* ctrs, int hc_iter, int dim, int n_ctr, int coor_dim)
{
    Eigen::VectorXd v(2);

    for (int i = 0; i < n_ctr; i++) {
        int c = 0;
        for (int j = 0; j < dim; j++) {
            v << coor[2*j]-ctrs[2*i], coor[2*j+1]-ctrs[2*i+1];
            if ( v.lpNorm<Eigen::Infinity>() <= 0.5 * width + Eigen::NumTraits<double>::epsilon() ) {
                indices[i*(coor_dim / n_ctr) + c] = j;
                indices_inv[j] = i;
                c += 1;
            }
            if ( c == coor_dim / n_ctr )
                break;
        }
        assert(c == coor_dim / n_ctr);
    }

    return;
}

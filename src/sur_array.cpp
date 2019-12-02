#include "sur_array.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include <ctime>
#include <math.h>

#include <eigen3/Eigen/Dense>

void c_compute_sur_omega(int rows, int cols, double* omega, double* phi, double* gamma, int vc)
{
    for (int i = 0; i < cols; i++) {
        // Determine row index of the biggest entry in i th gamma column
        if(vc) {
          int temp_max = -1;
          for (int j = i; j < rows*cols; j += cols) {
            if ((i >= 1 && gamma[j] - phi[j - 1] > 0.) || (i == 0 && gamma[j] > 0.)) {
              if (temp_max == -1)
                temp_max = j;
              else if (gamma[j] > gamma[temp_max])
                temp_max = j;
            }
          }
          if (temp_max == -1)
            std::cout << "ERROR!" << std::endl;
          omega[temp_max] = 1;
        }
        else {
          int temp_max = i;
          for (int j = i; j < rows*cols; j += cols) {
              if (gamma[j] > gamma[temp_max])
                  temp_max = j;
          }
          omega[temp_max] = 1;
        }
        // Update the i th column of phi
        for (int j = i; j < rows*cols; j += cols) {
            phi[j] = gamma[j] - omega[j];
        }

        // Update the i+1 th column of gamma
        if (i < cols-1){
            for (int j = i+1; j < rows*cols; j += cols) {
                gamma[j] += phi[j-1];
            }
        }
    }

    return;
}

void c_compute_median_filter(int hc_iter, double* vec, double* ctrs, int lenCtrs, double* avg, double width, int* idx_table, int len1IdxTable, int len2IdxTable)
{
    int n_window_cells = 9;
    int n_squares = 1 << hc_iter;

    Eigen::VectorXd i2_neighborhood = Eigen::ArrayXd::Zero(n_squares*n_squares*n_window_cells);
    Eigen::VectorXd ctrs2pair = Eigen::ArrayXd::Zero(lenCtrs);
    for ( int i = 0; i < lenCtrs; i++ ) {
        ctrs2pair[i] = ctrs[i] / width - 0.5;
    }
    Eigen::VectorXi indices = Eigen::ArrayXi::Zero(lenCtrs/2);
    for ( int i = 0; i < lenCtrs/2; i++ ) {
        indices[i] = round(ctrs2pair[2*i] * n_squares + ctrs2pair[2*i+1]);
    }

    Eigen::VectorXi indices_nb_map = Eigen::ArrayXi::Zero(lenCtrs/2 * n_window_cells);
    for ( int i = 0; i < lenCtrs/2; i++ ) {
        indices_nb_map[n_window_cells*i]   = indices[i] - n_squares - 1;
        indices_nb_map[n_window_cells*i+1] = indices[i] - n_squares;
        indices_nb_map[n_window_cells*i+2] = indices[i] - n_squares + 1;
        indices_nb_map[n_window_cells*i+3] = indices[i] - 1;
        indices_nb_map[n_window_cells*i+4] = indices[i];
        indices_nb_map[n_window_cells*i+5] = indices[i] + 1;
        indices_nb_map[n_window_cells*i+6] = indices[i] + n_squares - 1;
        indices_nb_map[n_window_cells*i+7] = indices[i] + n_squares;
        indices_nb_map[n_window_cells*i+8] = indices[i] + n_squares + 1;
    }

    for ( int i = 0; i < lenCtrs/2; i++ ) {
        if ( (ctrs2pair[2*i] == 0) | (ctrs2pair[2*i+1] == 0) )
            indices_nb_map[n_window_cells*i]   = indices[i];
        if ( ctrs2pair[2*i] == 0 )
            indices_nb_map[n_window_cells*i+1] = indices[i];
        if ( (ctrs2pair[2*i] == 0) | (ctrs2pair[2*i+1] == n_squares - 1) )
            indices_nb_map[n_window_cells*i+2] = indices[i];
        if ( ctrs2pair[2*i+1] == 0 )
            indices_nb_map[n_window_cells*i+3] = indices[i];
        if ( ctrs2pair[2*i+1] == n_squares - 1 )
            indices_nb_map[n_window_cells*i+5] = indices[i];
        if ( (ctrs2pair[2*i] == n_squares - 1) | (ctrs2pair[2*i+1] == 0) )
            indices_nb_map[n_window_cells*i+6] = indices[i];
        if ( ctrs2pair[2*i] == n_squares - 1 )
            indices_nb_map[n_window_cells*i+7] = indices[i];
        if ( (ctrs2pair[2*i] == n_squares - 1) | (ctrs2pair[2*i+1] == n_squares - 1) )
            indices_nb_map[n_window_cells*i+8] = indices[i];
    }

    for ( int i = 0; i < lenCtrs/2 * n_window_cells; i++ ) {
        i2_neighborhood[indices_nb_map[i]*n_window_cells + i % n_window_cells] = avg[int(floor(i/n_window_cells))];
    }
    for ( int i = 0; i < lenCtrs/2; i++ ) {
        std::sort(i2_neighborhood.data() + i*n_window_cells, i2_neighborhood.data() + (i+1)*n_window_cells, std::less<double>());
    }

    Eigen::VectorXd med_values = Eigen::ArrayXd::Zero(lenCtrs/2);
    for ( int i = 0; i < lenCtrs/2; i++ ) {
        med_values[i] = i2_neighborhood[indices[i]*n_window_cells+4];
    }

    for ( int j = 0; j < len1IdxTable; j++ ) {
        for ( int i = 0; i < len2IdxTable; i++ ) {
            vec[idx_table[j*len2IdxTable+i]] = med_values[j];
        }
    }
}

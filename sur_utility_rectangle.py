import copy
import numpy as np

from dolfin import *

import hilbertcurve.hilbertcurve as hilbert
import hilbert_cython
import sur_array_cython


def u2alpha(u_bangs, u):
    assert u.ndim == 1
    assert u_bangs.ndim == 1
    assert np.linalg.norm(np.sort(u_bangs) - u_bangs) == 0.0

    u = copy.deepcopy(u)
    u = np.maximum(u, u_bangs[0] * np.ones(u.shape))
    u = np.minimum(u, u_bangs[-1] * np.ones(u.shape))

    ubu_diff = np.tile(u_bangs[:, np.newaxis], (1, u.shape[0])) - u[np.newaxis, :]
    min_idx = np.argmin(np.abs(ubu_diff), axis=0)

    u_bangs_ext = np.zeros(u_bangs.shape[0] + 2)
    u_bangs_ext[0] = u_bangs[0] - 2. * np.max(u_bangs[1:] - u_bangs[:-1])
    u_bangs_ext[1:-1] = u_bangs
    u_bangs_ext[-1] = u_bangs[-1] + 2. * np.max(u_bangs[1:] - u_bangs[:-1])
    choice = np.sign(u - u_bangs[min_idx]).astype(int)
    lbd = np.zeros(u.shape[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        lbd = np.abs(np.nan_to_num((u - u_bangs[min_idx]) / (u_bangs[min_idx + choice] - u_bangs[min_idx])))
    assert np.linalg.norm((1. - lbd) * u_bangs[min_idx] + lbd * u_bangs[min_idx + choice] - u) < np.finfo(float).eps

    alpha = np.zeros((u_bangs.shape[0], u.shape[0]))
    alpha[min_idx + choice, range(u.shape[0])] = lbd
    alpha[min_idx, range(u.shape[0])] = 1. - lbd

    assert np.linalg.norm(alpha.transpose().dot(u_bangs) - u) < np.finfo(float).eps
    return alpha


class HcRectangle:
    def __init__(self, max_iter, x0, x1, use_cython=True):
        assert x0[0] < x1[0] and x0[1] < x1[1]
        self.max_iter, self.x0, self.x1 = max_iter, x0, x1
        self.mesh_width, self.mesh_height, self.dim = self.x1[0] - self.x0[0], self.x1[1] - self.x0[1], 2
        self.px0, self.px1 = Point(x0[0], x0[1]), Point(x1[0], x1[1])        
        self.cell_size = np.zeros((self.max_iter, self.dim), dtype=np.float64)
        cell_size_temp = np.zeros(self.max_iter, dtype=np.float64)
        
        self.lo_mesh = []
        self.lo_DG = []
        self.lo_ctrs = []
        self.lo_coor = []
        
        for i in range(1, max_iter + 1):
            self.lo_mesh += [RectangleMesh(self.px0, self.px1, 2**i, 2**i)]
            self.lo_DG += [FunctionSpace(self.lo_mesh[i - 1], 'DG', 0)]

            # Compute centers            
            n_ctrs = 2 ** (i * self.dim)
            if use_cython:           # Via Cython:
                ctrs = np.zeros((n_ctrs, 2), dtype=np.float64)
                ctrs, cell_size_temp = hilbert_cython.compute_ctrs(ctrs, cell_size_temp, i)
            else:                   # Via Python:
                ctrs, cell_size_temp[i - 1] = self.compute_ctrs(i)
            ctrs[:, 0] = ctrs[:, 0] * self.mesh_width + self.x0[0]
            ctrs[:, 1] = ctrs[:, 1] * self.mesh_height + self.x0[1]
            self.lo_ctrs += [ctrs]
            self.lo_coor += [self.lo_DG[i - 1].tabulate_dof_coordinates().reshape(self.lo_DG[i - 1].dim(), self.dim).astype(np.float64)]
        
        self.cell_size[:, 0] = cell_size_temp * self.mesh_width
        self.cell_size[:, 1] = cell_size_temp * self.mesh_height
        
        self.triangle_dict = {}
        for i in range(0, max_iter):
            f = Function(self.lo_DG[i])
            f.vector().set_local(np.linspace(0, self.lo_DG[i].dim() - 1, self.lo_DG[i].dim()))
            for j in range(i + 1, max_iter):
                fn = project(f, self.lo_DG[j])                
                fn_mat = np.zeros((self.lo_DG[j].dim(), 2), dtype=np.int32)
                fn_mat[:, 0] = np.arange(self.lo_DG[j].dim())
                self.triangle_dict[np.int32(j), np.int32(i)] = np.rint(fn.vector().get_local()).astype(np.int32)
                fn_mat[:, 1] = np.rint(fn.vector().get_local())
                fn_mat = fn_mat[fn_mat[:, 1].argsort()]
                self.triangle_dict[np.int32(i), np.int32(j)] = np.reshape(fn_mat[:, 0], (self.lo_DG[i].dim(), 2**(self.dim*(j - i))))
                
        self.square_to_triangle_dict = {}        
        self.triangle_to_square_dict = {}
        
        self.square_to_triangle_dict[(np.int32(0), np.int32(0))], self.triangle_to_square_dict[(np.int32(0), np.int32(0))] = \
            self.compute_idx_table_slow(1, self.lo_DG[0], self.lo_coor[0], self.lo_DG[0].dim())
        
        for i in range(1, max_iter):
            n_triangles = self.lo_DG[i].dim()
            n_squares = np.rint(2**(self.dim * (i + 1))).astype(np.int32)
            assert n_triangles % n_squares == 0

            idt_inv = np.zeros((n_triangles,), dtype=np.int32)
            
            sqr2sqr_prev = np.arange(2**(self.dim * i)).repeat(2**self.dim)
            triangle_options_prev = self.square_to_triangle_dict[(np.int32(i - 1), np.int32(i - 1))][sqr2sqr_prev]
            n_triangles_prev = triangle_options_prev.shape[1]
            triangles_i = self.triangle_dict[(np.int32(i - 1), np.int32(i))]
            triangle_options = triangles_i[triangle_options_prev.reshape(n_squares * triangle_options_prev.shape[1])]
            n_triangle_options_per_sqr = n_triangles_prev * triangles_i.shape[1]
            triangle_options_reshaped = triangle_options.reshape(n_squares * n_triangle_options_per_sqr)
            
            sqr_to_triangle_coor_options = np.zeros((self.dim, n_squares, n_triangle_options_per_sqr))
            sqr_to_triangle_indicator = np.zeros((self.dim, n_squares, n_triangle_options_per_sqr))
            sqr_to_triangle_ind = np.zeros((self.dim, n_squares, n_triangle_options_per_sqr))
            
            for j in range(self.dim):
                sqr_to_triangle_coor_options[j,:,:] = self.lo_coor[i][triangle_options_reshaped,j].reshape((n_squares, n_triangle_options_per_sqr))
                ctrs_j = np.tile(self.lo_ctrs[i][:, j], (n_triangle_options_per_sqr, 1)).transpose()
                sqr_to_triangle_indicator[j,:,:] = np.abs(sqr_to_triangle_coor_options[j,:,:] - ctrs_j) \
                    <= .5 * self.cell_size[i,j] + np.finfo(float).eps
                if j == 0:
                    sqr_to_triangle_ind = sqr_to_triangle_indicator[j,:,:]
                else:
                    sqr_to_triangle_ind = np.logical_and(sqr_to_triangle_ind, sqr_to_triangle_indicator[j,:,:])
            
            idt = triangle_options.reshape((n_squares, n_triangle_options_per_sqr))[np.where(sqr_to_triangle_ind == True)].reshape(
                (n_squares, np.rint(n_triangles / n_squares).astype(np.int32)))
            
            self.square_to_triangle_dict[(np.int32(i), np.int32(i))] = idt 
            idt_inv = np.repeat(np.rint(np.arange(n_squares)).astype(np.int32), 2)[np.argsort(idt.reshape(n_triangles))]            
            self.triangle_to_square_dict[(np.int32(i), np.int32(i))] = idt_inv
        
    def compute_ctrs(self, hc_iter):
        width = 2. ** (-hc_iter)
        n_ctr = 2 ** (hc_iter * self.dim)
        hc = hilbert.HilbertCurve(hc_iter, self.dim)
        ctrs = np.zeros((n_ctr, 2), dtype=np.float64)
        for i in range(n_ctr):
            ctrs[i] = hc.coordinates_from_distance(i)
        ctrs *= width
        ctrs += .5 * width
        return ctrs, width

    def compute_idx_table_slow(self, hc_iter, DG, coor, n_dof):
        n_ctr = self.lo_ctrs[hc_iter - 1].shape[0]
        assert coor.shape[0] % n_ctr == 0
        indices = np.zeros((n_ctr, int(coor.shape[0] / n_ctr)), dtype=np.int32)
        indices_inv = np.zeros(coor.shape[0], dtype=np.int32)
        for i in range(n_ctr):
            ctr = self.lo_ctrs[hc_iter - 1][i]
            c = 0
            for j in range(n_dof):
                if np.abs(coor[j, 0] - ctr[0]) <= .5 * self.cell_size[hc_iter - 1, 0] + np.finfo(float).eps \
                    and np.abs(coor[j, 1] - ctr[1]) <= .5 * self.cell_size[hc_iter - 1, 1] + np.finfo(float).eps:
                    indices[i, c] = j
                    indices_inv[j] = i
                    c = c + 1
                if c == coor.shape[0] / n_ctr:
                    break
            assert c == coor.shape[0] / n_ctr
        return indices, indices_inv

    def compute_fun_avg(self, fun_in, hc_iter_in, hc_iter_avg):
        fun = project(fun_in, self.lo_DG[hc_iter_in - 1])
        assert hc_iter_avg <= hc_iter_in
        avg_triangle = np.mean(fun.vector().get_local()[self.triangle_dict[(np.int32(hc_iter_avg - 1), np.int32(hc_iter_in - 1))]], axis=1) \
            if hc_iter_avg < hc_iter_in else fun.vector().get_local()
        avg_square = np.mean(avg_triangle[self.square_to_triangle_dict[(np.int32(hc_iter_avg - 1), np.int32(hc_iter_avg - 1))]], axis=1)
        fun = Function(self.lo_DG[hc_iter_avg - 1])
        avg_triangle = avg_square[self.triangle_to_square_dict[(np.int32(hc_iter_avg - 1), np.int32(hc_iter_avg - 1))]]
        fun.vector().set_local(avg_triangle)
        return avg_square, fun

    def scatter_to_fun(self, mat, hc_iter_in, hc_iter_out):
        assert hc_iter_in <= hc_iter_out
        triangle_vals = mat[self.triangle_to_square_dict[(np.int32(hc_iter_in - 1), np.int32(hc_iter_in - 1))]]
        if hc_iter_in < hc_iter_out:
            triangle_vals = triangle_vals[self.triangle_dict[(np.int32(hc_iter_out - 1), np.int32(hc_iter_in - 1))]]
        fun = Function(self.lo_DG[hc_iter_out - 1])
        fun.vector().set_local(triangle_vals)
        return fun

    def compute_sur_omega(self, alpha, use_cython=True):
        n_elem = alpha.shape[1]
        omega = np.zeros(alpha.shape)
        phi = np.zeros(alpha.shape)
        gamma = copy.deepcopy(alpha)
        if use_cython:
            sur_array_cython.compute_sur_omega(alpha.shape[0], alpha.shape[1], omega, phi, gamma)
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

    def compute_sur_omega_permutation(self, alpha, use_cython=True):
        n_elem = alpha.shape[1]
        permutation = np.random.permutation(np.arange(n_elem))
        permutation_inv = np.argsort(permutation)
        alpha_mod = copy.deepcopy(alpha[:, permutation])
        assert np.allclose(alpha_mod[:, permutation_inv], alpha)

        omega = np.zeros(alpha.shape)
        phi = np.zeros(alpha.shape)
        gamma = copy.deepcopy(alpha_mod)

        omega = omega.copy(order='C')
        phi = phi.copy(order='C')
        gamma = gamma.copy(order='C')

        if use_cython:
            sur_array_cython.compute_sur_omega(alpha.shape[0], alpha.shape[1], omega, phi, gamma)
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

        omega = omega[:, permutation_inv]
        gamma = gamma[:, permutation_inv]
        phi = phi[:, permutation_inv]
        return omega, gamma, phi

    def compute_sur_permutation(self, lo_alpha_funs, hc_iter, hc_iter_out, use_cython=True):
        alpha_arrs = list(map(lambda fun: self.compute_fun_avg(fun, hc_iter, hc_iter)[0], lo_alpha_funs))
        omega_arrs, gamma_arrs, phi_arrs = self.compute_sur_omega_permutation(np.array(alpha_arrs), use_cython)
        alpha_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), np.array(alpha_arrs)))
        omega_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), list(omega_arrs)))
        phi_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), list(phi_arrs)))
        return alpha_funs, omega_funs, phi_funs

    def compute_sur(self, lo_alpha_funs, hc_iter, hc_iter_out, use_cython=True):
        alpha_arrs = list(map(lambda fun: self.compute_fun_avg(fun, hc_iter, hc_iter)[0], lo_alpha_funs))
        omega_arrs, gamma_arrs, phi_arrs = self.compute_sur_omega(np.array(alpha_arrs), use_cython)
        alpha_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), np.array(alpha_arrs)))
        omega_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), list(omega_arrs)))
        phi_funs = list(map(lambda avg: self.scatter_to_fun(avg, hc_iter, hc_iter_out), list(phi_arrs)))
        return alpha_funs, omega_funs, phi_funs
    
    def compute_median_filter(self, hc_iter, fun, hc_iter_out, use_cython=False):
        avg, _ = self.compute_fun_avg(fun, hc_iter, hc_iter)        
        
        vec = np.zeros(self.lo_DG[hc_iter - 1].dim())
        ctrs2pair = copy.deepcopy(self.lo_ctrs[hc_iter - 1])
        for i in range(self.dim):
            ctrs2pair[:, i] -= self.x0[i]
            ctrs2pair[:, i] /= self.cell_size[hc_iter - 1, i] 
            ctrs2pair[:, i] -= .5
        
        n_window_cells = 9
        
        n_squares = 2**hc_iter
        i2_neighborhood = np.zeros((n_squares**2, n_window_cells))
        indices = np.rint(ctrs2pair[:, 0] * n_squares + ctrs2pair[:, 1]).astype(np.int32)
        
        indices_nb_map = np.zeros((indices.shape[0], n_window_cells), dtype=np.int32)
        indices_nb_map[:, 0] = (indices - n_squares - 1).astype(np.int32)
        indices_nb_map[:, 1] = (indices - n_squares).astype(np.int32)
        indices_nb_map[:, 2] = (indices - n_squares + 1).astype(np.int32)
        indices_nb_map[:, 3] = (indices - 1).astype(np.int32)
        indices_nb_map[:, 4] = indices.astype(np.int32)
        indices_nb_map[:, 5] = (indices + 1).astype(np.int32)
        indices_nb_map[:, 6] = (indices + n_squares - 1).astype(np.int32)
        indices_nb_map[:, 7] = (indices + n_squares).astype(np.int32)
        indices_nb_map[:, 8] = (indices + n_squares + 1).astype(np.int32)

        idx_out_of_bounds = np.where((ctrs2pair[:, 0] == 0) | (ctrs2pair[:, 1] == 0))
        indices_nb_map[idx_out_of_bounds, 0] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where(ctrs2pair[:, 0] == 0)
        indices_nb_map[idx_out_of_bounds, 1] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where((ctrs2pair[:, 0] == 0) | (ctrs2pair[:, 1] == n_squares - 1))
        indices_nb_map[idx_out_of_bounds, 2] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where(ctrs2pair[:, 1] == 0)
        indices_nb_map[idx_out_of_bounds, 3] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where(ctrs2pair[:, 1] == n_squares - 1)
        indices_nb_map[idx_out_of_bounds, 5] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where((ctrs2pair[:, 0] == n_squares - 1) | (ctrs2pair[:, 1] == 0))
        indices_nb_map[idx_out_of_bounds, 6] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where(ctrs2pair[:, 0] == n_squares - 1)
        indices_nb_map[idx_out_of_bounds, 7] = indices[idx_out_of_bounds]
        idx_out_of_bounds = np.where((ctrs2pair[:, 0] == n_squares - 1) | (ctrs2pair[:, 1] == n_squares - 1))
        indices_nb_map[idx_out_of_bounds, 8] = indices[idx_out_of_bounds]
        
        idx_pairs = (indices_nb_map.reshape(indices.shape[0] * n_window_cells).astype(np.int32),
                     np.tile(np.arange(n_window_cells), indices.shape[0]).astype(np.int32))
        
        i2_neighborhood[idx_pairs] = np.repeat(avg, n_window_cells)
        i2_neighborhood.sort(axis=1)
        med_values = i2_neighborhood[indices.astype(np.int32), 4]
        
        return self.scatter_to_fun(med_values, hc_iter, hc_iter_out)

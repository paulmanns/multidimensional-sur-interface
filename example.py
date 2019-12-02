import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy as sp
from scipy import ndimage
from time import time as time_alt

from dolfin import *

import sur_utility_rectangle

PLOT_CONTROL_CONV=True
PLOT_STATE_CONV=True

arr = imageio.imread('./img/hilbert.png')

assert arr.shape[0]**-1 == arr.shape[1]**-1
img_width = arr.shape[0]**-1

print("Size reduced for computers with less memory and CPU power.")
K = 16
gray = 1. - np.flipud(np.fliplr(np.rot90(arr[::K, ::K] / 255.)))

# =============
# Hilbert Curve
# =============
hc_max_iter = np.round(np.log2(gray.shape[0])).astype(np.int32)
t1 = time_alt()
N = int(arr.shape[0] / K)
x0, x1 = np.array([0., 0.]), np.array([1., 1.])
hc = sur_utility_rectangle.HcRectangle(hc_max_iter, x0, x1, True)
t2 = time_alt()
print('Total elapsed time to generate Hilbert curve mappings: ', t2 - t1)

# =============
# Mesh
# =============
mesh = hc.lo_mesh[-1]
LG = FunctionSpace(mesh, "CG", 1)
DG = FunctionSpace(mesh, "DG", 0)

# =============
# Boundary
# =============
def u0_bnd(x, on_bnd):
    return on_bnd

u0 = Expression("0", degree=2)
bc = DirichletBC(LG, u0, u0_bnd)

# =============
# Problem
# =============
u = TrialFunction(LG)
v = TestFunction(LG)

# == Laplace with homogeneous Dirichlet boundary ==
A_weak_form = dot(grad(u), grad(v)) * dx

f_alpha = Function(DG)
coor = DG.tabulate_dof_coordinates().reshape(DG.dim(), mesh.geometry().dim())
coor_int = np.floor(coor * N).astype(int)
lo_coor_int = [tuple(a) for a in coor_int]
a = np.zeros(f_alpha.vector().get_local().shape)
for i, coor in enumerate(lo_coor_int):
    a[i] = gray[coor]
f_alpha.vector().set_local(a)
L = f_alpha * v * dx

a_zero = Function(DG)
a_zero.interpolate(Constant(1.))
a_zero = project(a_zero - f_alpha, DG)

y_alpha = Function(LG)
solve(A_weak_form == L, y_alpha, bc)

lo_omega_sur = []
lo_y_sur = []

n_permutations = 5
mo_norm = np.zeros((hc_max_iter, n_permutations))

for hc_iter in range(1, hc_max_iter+1):
    print('SUR on grid #', hc_iter)
    for i in range(n_permutations):
        # ==============
        # SUR
        # ==============
        lo_alpha_funs = [f_alpha, a_zero]
        if i == 0:
            # For Hilbert curve-induced cell odering, we usually observe grid cell volume as convergence rate for the state approximation error.
            alpha_funs, omega_funs, phi_funs = hc.compute_sur(lo_alpha_funs, hc_iter, hc_max_iter, True)
        else:
            # For random cell ordering, we usually observe SQRT(grid cell volume) as convergence rate for the state approximation error.
            alpha_funs, omega_funs, phi_funs = hc.compute_sur_permutation(lo_alpha_funs, hc_iter, hc_max_iter, True)
        y = Function(LG)
        solve(A_weak_form == omega_funs[0] * v * dx, y, bc)
        if i == 0:
            lo_y_sur += [y]
            lo_omega_sur += [omega_funs[0]]
        mo_norm[hc_iter - 1, i] = norm(project(y_alpha - y, LG), 'L2')

np.savetxt('./out/permutation_stats.txt', mo_norm, delimiter=' ')

print("State error for Hilbert curve orderings over the iterations.")
for i in range(len(lo_y_sur)):
    print('%2d L2 ' % i, norm(project(y_alpha - lo_y_sur[i], LG), 'L2'))
    print('%2d H1 ' % i, norm(project(y_alpha - lo_y_sur[i], LG), 'H1'))
print()

vmin = np.min(lo_y_sur[-1].vector().get_local())
vmax = np.max(lo_y_sur[-1].vector().get_local())
for i in range(hc_max_iter):
    vmin = np.min(np.array([np.min(lo_y_sur[i].vector().get_local()),
                            vmin]))
    vmax = np.max(np.array([np.max(lo_y_sur[i].vector().get_local()),
                            vmax]))

if PLOT_CONTROL_CONV:
    for i in range(hc_max_iter):
        print('Plotting control, iter %d ...' % (1 + i))
        plot(lo_omega_sur[i], title=r'$v(\omega^{(%d)})$' % (i + 1), vmin=0.0, vmax=1.0, cmap=cm.binary)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.show()
        plt.savefig('./out/ctrl_%d_%d_sur.png' % (hc_max_iter, 1 + i),
                    dpi=300, bbox_inches="tight", pad_inches=0)

if PLOT_STATE_CONV:
    for i in range(hc_max_iter):
        print('Plotting state, iter %d ...' % (1 + i))
        plot(lo_y_sur[i], title=r'$y(\omega^{(%d)})$' % (1 + i), vmin=vmin, vmax=vmax, cmap=cm.coolwarm)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.savefig('./out/state_%d_%d_sur.png' % (hc_max_iter, 1 + i),
                    dpi=300, bbox_inches="tight", pad_inches=0)



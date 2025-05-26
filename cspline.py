import numpy as np
from scipy.interpolate import interpn, interp1d
import matplotlib.pyplot as plt

def spline3d(points, values, xi):
    """
    3D cubic spline interpolation, first 2D on each z-slice then 1D along z.
    Args:
        points: (x, y, z) where x, y, z are 1D arrays of grid points
        values: ndarray, shape (Nx, Ny, Nz)
        xi: ndarray, shape (M, 3)，each row (x, y, z)
    Returns:
        ndarray, shape (M,)
    """
    x_grid, y_grid, z_grid = points
    _, _, Nz = values.shape
    xi = np.asarray(xi)
    M = xi.shape[0]
    
    # 2D spline interpolation for each z-slice
    interp2d_vals = np.zeros((Nz, M))
    for iz, _ in enumerate(z_grid):
        interp2d_vals[iz, :] = interpn((x_grid, y_grid), values[:, :, iz], xi[:, 0:2],
                                       method='splinef2d', bounds_error=False, fill_value=0)

    # 1D spline interpolation along z
    out = np.zeros(M)
    for i in range(M):
        fz = interp1d(z_grid, interp2d_vals[:, i], kind='cubic', bounds_error=False, fill_value=0)
        out[i] = fz(xi[i, 2])
    return out

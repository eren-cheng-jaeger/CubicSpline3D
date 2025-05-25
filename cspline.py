import numpy as np
from scipy.interpolate import interpn, interp1d
import matplotlib.pyplot as plt

def spline3d(points, values, xi):
    """
    3D cubic spline interpolation (Matlab风格), 先对每个z-slice做2D样条，再对z方向做1D样条。
    Args:
        points: (x, y, z) 3个1D数组
        values: ndarray, shape (Nx, Ny, Nz)
        xi: ndarray, shape (M, 3)，每行是 (x, y, z)
    Returns:
        ndarray, shape (M,)
    """
    x_grid, y_grid, z_grid = points
    _, _, Nz = values.shape
    xi = np.asarray(xi)
    M = xi.shape[0]
    # 先对每个z-slice做2D样条
    interp2d_vals = np.zeros((Nz, M))
    for iz, _ in enumerate(z_grid):
        # 对每个z层，做2D样条
        interp2d_vals[iz, :] = interpn((x_grid, y_grid), values[:, :, iz], xi[:, 0:2],
                                       method='splinef2d', bounds_error=False, fill_value=0)
    # 再对z方向做1D三次样条
    out = np.zeros(M)
    for i in range(M):
        fz = interp1d(z_grid, interp2d_vals[:, i], kind='cubic', bounds_error=False, fill_value=0)
        out[i] = fz(xi[i, 2])
    return out
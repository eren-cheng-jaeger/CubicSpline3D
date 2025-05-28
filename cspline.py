import numpy as np
from scipy.interpolate import interpn, interp1d

def spline3d(grid, values, pts):
    """
    3D cubic spline interpolation, first 2D on each z-slice then 1D along z.
    Args:
        grid: (x, y, z) where x, y, z are 1D arrays of grid points
        values: ndarray, shape (Nx, Ny, C)
        pts: ndarray, shape (N, 3)，each row (x, y, z)
    Returns:
        ndarray, shape (N,)
    """
    x_grid, y_grid, z_grid = grid
    _, _, C = values.shape
    pts = np.asarray(pts)
    N = pts.shape[0]
    
    # Check if Z dimension is insufficient for 3D cubic spline (need at least 4 points)
    if C <= 3: # 1 original + 2 padding = 3, or less
        # Use 2D spline interpolation instead
        # Extract the middle Z slice (original data is at index 1 due to padding)
        values_2d = values[:, :, 1]
        pts_2d = pts[:, :2]
        return interpn((x_grid, y_grid), values_2d, pts_2d, method='splinef2d', bounds_error=False, fill_value=0)

    # 2D spline interpolation for each z-slice
    interp2d_vals = np.zeros((C, N))
    for iz, _ in enumerate(z_grid):
        # All interpolation points for all z-slices (C N)
        interp2d_vals[iz, :] = interpn((x_grid, y_grid), values[:, :, iz], pts[:, 0:2],
                                       method='splinef2d', bounds_error=False, fill_value=0)

    # 1D spline interpolation along z
    out = np.zeros(N)
    for i in range(N):
        # Interpolation function for the i-th point
        fz = interp1d(z_grid, interp2d_vals[:, i], kind='cubic', bounds_error=False, fill_value=0)
        out[i] = fz(pts[i, 2])
    return out

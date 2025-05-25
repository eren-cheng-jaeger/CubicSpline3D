# CubicSpline3D

3D cubic spline interpolation for regular grids.  
This implementation uses the classic approach of "2D spline on each z-slice, then 1D cubic spline along z", with an interface consistent with `scipy.interpolate.interpn`.

## Files

- `cspline.py`  
  Provides the `spline3d(points, values, xi)` function for 3D cubic spline interpolation.

- `test_spline3d.ipynb` or test script  
  Includes tests for interpolation accuracy, comparison with `interpn` linear interpolation, and visualization.

## Usage Example

```python
import numpy as np
from cspline import spline3d

# Construct a regular grid
x = np.arange(4)
y = np.arange(4)
z = np.arange(5)
grid = (x, y, z)
data = np.random.rand(4, 4, 5)

# Interpolation points (M, 3)
xi = np.array([
    [1.2, 2.5, 3.1],
    [0.5, 1.0, 2.0]
])

# 3D cubic spline interpolation
vals = spline3d(grid, data, xi)
print(vals)
```

## Testing & Visualization

- Supports accuracy tests at known points, interpolation points, and out-of-bounds points
- Supports comparison with `scipy.interpolate.interpn` using the `linear` method
- Supports slice heatmaps and 1D curve smoothness comparison

## Notes

- `splinef2d` requires more than 3 points in each direction (at least 4).
- Only suitable for regular grid data.
- The 3D interpolation is a "2D spline + 1D spline" combination, not a true global 3D spline.

## References

- [scipy.interpolate.interpn documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html)

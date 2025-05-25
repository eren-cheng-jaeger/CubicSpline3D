from scipy.interpolate import interpn
from .cspline import spline3d
import numpy as np
import matplotlib.pyplot as plt

def test_spline3d():
    print("=== Testing spline3d (2D+1D cubic, x-y-z order) ===")

    # 构造长方体网格 (4,4,5)
    x = np.arange(4)
    y = np.arange(4)
    z = np.arange(5)
    grid = (x, y, z)
    data = np.arange(4*4*5).reshape(4, 4, 5).astype(float)

    # 测试整数点
    test_points = np.array([
        [0, 0, 0],
        [3, 3, 4],
        [1, 2, 3],
        [2, 0, 1]
    ])
    expected = [data[0,0,0], data[3,3,4], data[1,2,3], data[2,0,1]]
    vals_spline = spline3d(grid, data, test_points)
    vals_linear = interpn(grid, data, test_points, method='linear', bounds_error=False, fill_value=0)
    print("Known points (spline3d vs interpn-linear):")
    for i, v in enumerate(vals_spline):
        print(f"Point {test_points[i]}: spline3d={v:.2f}, interpn-linear={vals_linear[i]:.2f}, expected={expected[i]:.2f}")

    # 测试插值点
    interp_points = np.array([
        [0.5, 0.5, 0.5],
        [1.0, 1.5, 2.0],
        [2.0, 3.0, 1.5]
    ])
    interp_vals_spline = spline3d(grid, data, interp_points)
    interp_vals_linear = interpn(grid, data, interp_points, method='linear', bounds_error=False, fill_value=0)
    print("Interpolated points (spline3d vs interpn-linear):")
    for pt, v1, v2 in zip(interp_points, interp_vals_spline, interp_vals_linear):
        print(f"Point {pt}: spline3d={v1:.2f}, interpn-linear={v2:.2f}")

    # 测试超界
    out_points = np.array([
        [-1, 0, 0],
        [0, 4, 0],
        [0, 0, 5]
    ])
    out_vals_spline = spline3d(grid, data, out_points)
    out_vals_linear = interpn(grid, data, out_points, method='linear', bounds_error=False, fill_value=0)
    print("Out-of-bounds points (spline3d vs interpn-linear):")
    for pt, v1, v2 in zip(out_points, out_vals_spline, out_vals_linear):
        print(f"Point {pt}: spline3d={v1:.2f}, interpn-linear={v2:.2f} (should be 0.0)")

    # 可视化 z=2 切片的 x-y 网格插值
    xx, yy = np.meshgrid(np.linspace(0, 2, 30), np.linspace(0, 3, 40), indexing='ij')
    zz = np.full_like(xx, 2.0)
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    vals_spline = spline3d(grid, data, pts).reshape(xx.shape)
    vals_linear = interpn(grid, data, pts, method='linear', bounds_error=False, fill_value=0).reshape(xx.shape)
    diff = vals_spline - vals_linear
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axs[0].imshow(vals_spline, origin='lower', aspect='auto', extent=[0,3,0,2])
    axs[0].set_title('spline3d z=2')
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(vals_linear, origin='lower', aspect='auto', extent=[0,3,0,2])
    axs[1].set_title('interpn-linear z=2')
    plt.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(diff, origin='lower', aspect='auto', extent=[0,3,0,2], cmap='bwr')
    axs[2].set_title('Difference (spline3d - linear)')
    plt.colorbar(im2, ax=axs[2])
    plt.tight_layout()
    plt.show()

    # 用非线性数据
    x = np.arange(4)
    y = np.arange(4)
    z = np.arange(5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    data = np.sin(X) + np.cos(Y) + Z**2

    # 1D切线对比
    z_line = np.linspace(0, 4, 100)
    pts = np.stack([np.full_like(z_line, 1.5), np.full_like(z_line, 2.0), z_line], axis=-1)
    vals_spline = spline3d((x, y, z), data, pts)
    vals_linear = interpn((x, y, z), data, pts, method='linear', bounds_error=False, fill_value=0)
    plt.plot(z_line, vals_spline, label='spline3d')
    plt.plot(z_line, vals_linear, label='linear')
    plt.legend()
    plt.title('Smoothness Comparison along z=1.5, y=2.0')
    plt.show()

    print("All tests and visualizations done!")

if __name__ == "__main__":
    test_spline3d()
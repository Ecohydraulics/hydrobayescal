"""Analytic stand-ins for solver responses, shared by several test modules."""
import numpy as np


def wall_layer(points, heights=(0.1, 0.4, 0.8), depth=0.3, bulk_velocity=0.5):
    """Log layer of a k-epsilon flume at fixed discharge: C_mu and k_s -> [u, k] at three heights, interleaved."""
    points = np.atleast_2d(points)
    cmu, ks = points[:, :1], points[:, 1:]
    ustar = 0.41 * bulk_velocity / (np.log(30.0 * depth / ks) - 1.0)
    z = np.asarray(heights) * depth
    out = np.empty((points.shape[0], 2 * len(heights)))
    out[:, 0::2] = ustar / 0.41 * np.log(30.0 * z / ks)
    out[:, 1::2] = ustar ** 2 / np.sqrt(cmu) * np.exp(-2.0 * z / depth)
    return out

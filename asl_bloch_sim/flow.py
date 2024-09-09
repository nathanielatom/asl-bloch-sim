import numpy as np
from scipy.interpolate import CubicSpline

def binormalize(arr, min, max, axis=None):
    minarr = arr.min(axis=axis)
    return (arr - minarr) * (max - min) / (arr.max(axis=axis) - minarr) + min

def integrate_trajectory(velocity, dt, position_offset=0):
    return np.cumsum(velocity) * dt - position_offset

def constant(start=0, stop=2, num=1000, velocity=200, **kwargs):
    x = np.linspace(start, stop, num)
    y = np.full(num, velocity)
    return x, y, integrate_trajectory(y, dt=x[1] - x[0], **kwargs)

def holdsworth_cca(start=0, stop=2, num=1000,
                   interbeat_interval=0.917, systolic_velocity=76, diastolic_velocity=30, **kwargs):
    # Characterization of common carotid artery blood-flow waveforms in normal human subjects
    # Holdsworth et al. 1999
    # https://doi.org/10.1088/0967-3334/20/3/301
    features_time = [0, 0.055, 0.110, 0.116, 0.153, 0.116 + 0.103, 0.398, 0.917] # s
    features_flow = [25.9, 20.9, 47.7, 64.6, 108.2, 64.6, 19.4, 25.9] # cm/s
    estimated_interpolated_point_time = [0.25, 0.3, 0.35, 0.48, 0.6, 0.7, 0.8] # s
    estimated_interpolated_point_flow = [48, 46, 38, 40, 31, 28, 27.5] # cm/s
    x = np.concatenate((features_time, estimated_interpolated_point_time))
    y = np.concatenate((features_flow, estimated_interpolated_point_flow))
    y = y[np.argsort(x)]
    x.sort()

    x *= interbeat_interval / x[-1]
    y = binormalize(y, diastolic_velocity, systolic_velocity)
    cs = CubicSpline(x, y, bc_type='periodic')

    interp_x = np.linspace(start, stop, num)
    waveform = cs(interp_x)
    return interp_x, waveform, integrate_trajectory(waveform, dt=interp_x[1] - interp_x[0], **kwargs)

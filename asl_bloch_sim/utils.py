import numpy as np

from asl_bloch_sim import xp, get_array_module

def expand_dims_to(arr1, arr2, dimodifier=0):
    """
    Expand dimensions of arr1 to be broadcastable with arr2.

    Parameters
    ----------
    arr1 : ndarray
        Array to expand after the last axis.
    arr2 : ndarray
        Array to match dimensions.
    dimodifier : int, optional
        Number of manual extra dimensions to expand. Default is 0.

    Returns
    -------
    ndarray
        Array with dimensions expanded.

    """
    xp = get_array_module(arr1, arr2)
    if not xp.isscalar(arr2):
        arr1 = xp.expand_dims(arr1, tuple(range(-arr2.ndim - dimodifier, 0)))
    return arr1

def has_time_axis(arr, signal, axis=0):
    xp = get_array_module(signal)
    try:
        return xp.shape(arr)[axis] == xp.shape(signal)[axis]
    except IndexError:
        return False

def dot(a, b, *, keepdims=False, axis=-1):
    """
    Compute the dot product of two arrays.

    Parameters
    ----------
    a : array_like
        First input array.
    b : array_like
        Second input array.
    keepdims : bool, optional
        If True, the output array will have the same number of dimensions as the input arrays, with size 1 in the reduced dimensions. Default is False.
    axis : int, optional
        Axis along which the dot product is computed. Default is -1.

    Returns
    -------
    ndarray
        Dot product of `a` and `b`.

    Raises
    ------
    ValueError: If the input arrays have incompatible shapes.

    Notes
    -----
    - If either `a` or `b` is a scalar, the dot product is computed as the element-wise multiplication of `a` and `b`.
    - The dot product is computed using the Einstein summation convention.

    Examples
    --------

    .. code-block:: python

        >>> a = [1, 2, 3]
        >>> b = [4, 5, 6]
        >>> dot(a, b)
        32

    """
    xp = get_array_module(a, b)
    if xp.isscalar(a) or xp.isscalar(b):
        return a * b

    return xp.sum(a * b, axis=axis, keepdims=keepdims)

def rodrigues_rotation(v, k, theta, *, normalize=True, axis=-1):
    """
    Apply Rodrigues rotation formula to rotate a vector `v` around an axis `k` by an angle `theta`.

    Parameters
    ----------
    v : array_like
        The vector to be rotated.
    k : array_like
        The rotation axis.
    theta : float
        The rotation angle in radians.
    normalize : bool, optional
        Whether to normalize the rotation axis `k`. Default is True.
    axis : int, optional
        The axis along which to compute the norm of `k`. Default is -1.

    Returns
    -------
    array_like
        The rotated vector.

    Notes
    -----
    The Rodrigues rotation formula is given by:

    .. math::

        v_{\\text{rot}} = v \\cos(\\theta) + (k \\times v) \\sin(\\theta) + k (k \\cdot v) (1 - \\cos(\\theta))

    where :math:`v_{\\text{rot}}` is the rotated vector, :math:`v` is the original vector, :math:`k` is the rotation axis,
    and :math:`\\theta` is the rotation angle.

    Examples
    --------
    >>> import numpy as np
    >>> from asl_bloch_sim import bloch
    >>> v = np.array([1, 0, 0])
    >>> k = np.array([0, 0, 1])
    >>> theta = np.pi / 2
    >>> bloch.rodrigues_rotation(v, k, theta)
    array([0., 1., 0.])

    """
    # once https://github.com/cupy/cupy/issues/7801 is resolved, we can use
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_rotvec.html
    # flatten extra dimensions to (N, 3), from scipy.spatial.transform import Rotation
    # Rotation.from_rotvec(k * theta).apply(v), unflatten extra dimensions

    xp = get_array_module(v, k, theta)
    if normalize:
        k = k / xp.linalg.norm(k, axis=axis, keepdims=True)
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)
    dot_kv = utils.dot(k, v, axis=axis, keepdims=True)
    cross_kv = xp.cross(k, v, axis=axis)
    v_rot = v * cos_theta + cross_kv * sin_theta + k * dot_kv * (1 - cos_theta)
    return v_rot

def binormalize(arr, min, max, axis=None):
    minarr = arr.min(axis=axis)
    return (arr - minarr) * (max - min) / (arr.max(axis=axis) - minarr) + min

def shift_complex_phase(waveform, phase, **kwargs):
    # shift the phase of a complex waveform
    # higher frequency components will be shifted less in time
    # a phase shift of pi is equivalent to multiplying by -1
    if phase == 0:
        return waveform
    phase_factor = np.exp(1j * phase)
    return np.fft.ifft(phase_factor * np.fft.fft(waveform, **kwargs), **kwargs)

def shift_overall_phase(waveform, phase, period, dt, axis=None):
    # shift the entire waveform in time corresponding to a phase shift
    # of the fundamental frequency, based on the provided period
    shift = np.round(phase * period / (2 * np.pi * dt)).astype(int)
    return np.roll(waveform, shift, axis=axis)

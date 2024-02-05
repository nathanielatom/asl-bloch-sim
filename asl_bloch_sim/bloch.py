import numpy as np
from numba import njit

GAMMA_BAR = 42.5759e6 # Gyromagnetic ratio (Hz/T)
GAMMA = 2 * np.pi * GAMMA_BAR # Gyromagnetic ratio (rads/T/s)

@njit(parallel=True)
def norm_axis_0(arr):
    return np.sqrt((arr**2).sum(axis=0))

@njit(parallel=True)
def dot_axis_0(a, b):
    if a.ndim == 1:
        return np.dot(a, b)
    elif a.ndim == 2:
        result = np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            result[i] = np.dot(a[:, i], b[:, i])
        return result
    elif a.ndim == 3:
        result = np.zeros(a.shape[1:])
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                result[i, j] = np.dot(a[:, i, j], b[:, i, j])
        return result
    elif a.ndim == 4:
        result = np.zeros(a.shape[1:])
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                for k in range(a.shape[3]):
                    result[i, j, k] = np.dot(a[:, i, j, k], b[:, i, j, k])
        return result
    else:
        raise ValueError("Array dimension must be 1, 2, 3, or 4.")

@njit
def cross_axis_0(a, b):
    """
    Compute the cross product of two vectors.

    Parameters
    ----------
    a : ndarray
        First vector. Axis along which the cross product is computed is 0.
    b : ndarray
        Second vector. Axis along which the cross product is computed is 0.

    Returns
    -------
    ndarray
        Cross product of a and b.

    Notes
    -----
    The cross product is defined as:

    a x b = (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)

    Examples
    --------
    >>> a = np.array([1, 0, 0])
    >>> b = np.array([0, 1, 0])
    >>> cross(a, b)
    array([0, 0, 1])

    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.empty((3, *a.shape[1:]), dtype=np.float64)

    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]

    return c

def cross_product_numexpr(a, b, axis=0):
    import numexpr as ne
    # move axis of a and b to the start
    a = np.moveaxis(a, axis, 0)
    b = np.moveaxis(b, axis, 0)
    ax, ay, az = a
    bx, by, bz = b
    out = np.array([ne.evaluate("ay * bz - az * by"),
                    ne.evaluate("az * bx - ax * bz"),
                    ne.evaluate("ax * by - ay * bx")])
    return out

def norm_numexpr(arr, axis=0):
    import numexpr as ne
    # move axis of a and b to the start
    arr = np.moveaxis(arr, axis, 0)
    ax, ay, az = arr
    out = ne.evaluate("sqrt(ax ** 2 + ay ** 2 + az ** 2)")
    return out

@njit(parallel=True)
def rodrigues_rotation_axis_0(v, k, theta):
    """
    Apply Rodrigues rotation formula to rotate a vector v around an axis k by
    an angle theta.

    Parameters
    ----------
    v : ndarray
        Input vector to be rotated. Axis along which the rotation is performed is 0.
    k : ndarray
        Axis of rotation. Axis along which the rotation is performed is 0.
    theta : float
        Angle of rotation in radians.

    Returns
    -------
    ndarray
        Rotated vector.

    Notes
    -----
    The Rodrigues rotation formula is given by:

    v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))

    where v_rot is the rotated vector, x denotes the cross product, and . denotes
    the dot product.

    Examples
    --------
    >>> v = np.array([1, 0, 0])
    >>> k = np.array([0, 0, 1])
    >>> theta = np.pi / 2
    >>> rodrigues_rotation(v, k, theta)
    array([0., 1., 0.])

    """
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    k = k / norm_axis_0(k)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    dot_kv = dot_axis_0(k, v)
    cross_kv = cross_axis_0(k, v)
    v_rot = v * cos_theta + cross_kv * sin_theta + k * dot_kv * (1 - cos_theta)
    return v_rot

def dot(a, b, *, keepdims=False, axis=-1):
    import string
    a_shape = string.ascii_letters[:a.ndim]
    out_shape = a_shape.replace(a_shape[axis], '')
    subscripts = f'{a_shape},{a_shape}->{out_shape}'
    out = np.einsum(subscripts, a, b)
    if keepdims:
        shap = list(a.shape)
        shap[axis] = 1
        return out.reshape(shap)
    return out

def rodrigues_rotation(v, k, theta, *, normalize=True, axis=-1):
    if normalize:
        k = k / np.linalg.norm(k, axis=axis, keepdims=True)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    dot_kv = dot(k, v, axis=axis, keepdims=True)
    cross_kv = np.cross(k, v, axis=axis)
    v_rot = v * cos_theta + cross_kv * sin_theta + k * dot_kv * (1 - cos_theta)
    return v_rot

def unit_field_and_angle(B_field, dt, tol=1e-14, axis=-1):
    """
    Compute the unit field vector and rotation angle for a given field vector
    and time step.

    Parameters
    ----------
    B_field : ndarray
        Magnetic field vector in Tesla.
    dt : float
        Time step in seconds.
    tol : float, optional
        Numerical tolerance to avoid division by zero. The
        angle of `B_field` vector with magnitude smaller than `tol` is
        set to 0. Default is 1e-14.
    axis : int, optional
        Axis along the field array which
        represents the 2D or 3D spatial field vector.
        Default is the last axis.

    Returns
    -------
    b : ndarray
        Unit field vector.
    ang : ndarray
        Rotation angle in radians. Same shape as B_field, except for
        the specified spatial axis.

    Notes
    -----
    The unit field vector and rotation angle are computed as:

    b = B / |B|
    ang = -γ * |B| * dt

    where |B| is the magnitude of the field vector, γ is the gyromagnetic
    ratio, and dt is the time step.

    """
    # Set length/magnitude of B
    B_length = np.linalg.norm(B_field, axis=axis, keepdims=True)
    mask = B_length <= tol
    B_length[mask] = tol

    # Set rotation-axis (unit-vector) components
    b_field = B_field / B_length

    # Set rotation-angle [rads], note negative sign for LHR
    ang = -GAMMA * B_length * dt
    ang[mask] = 0
    return b_field, ang

def precess(magnetization, B_field, dt, *, axis=-1, **kwargs):
    return rodrigues_rotation(magnetization, *unit_field_and_angle(B_field, dt, axis=axis, **kwargs),
                              normalize=False, axis=axis)

def relax(magnetization, T1, T2, dt, M0=(0, 0, 1),
          extend_shapes=True, match_shapes=True, axis=-1):
    """
    Apply relaxation to the magnetization vector.

    Parameters
    ----------
    magnetization : ndarray
        Magnetization vector, can be multidimensional.
    T1 : ndarray
        Longitudinal relaxation time in seconds.
    T2 : ndarray
        Transverse relaxation time in seconds.
    dt : float
        Time step in seconds.
    M0 : tuple, optional
        Initial magnetization vector. Default is (0, 0, 1), i.e. up
        along the z-axis.
    extend_shapes : bool, optional
        If True, the shapes of T1 and T2 are prefixed to the shape of the
        magnetization array. See `match_shapes`. Note when False, T1 and T2 should
        have the same shape as magnetization except along axis, which should be 1.
        Default is True.
    match_shapes : bool, optional
        If True, the shapes of T1 and T2 are not prefixed to the shape of the
        magnetization array when they match the start of its shape. Default is True.
    axis : int, optional
        Axis along the magnetization array which
        represents the 2D or 3D spatial magnetization vector.
        Default is the last axis.

    Returns
    -------
    updated_magnetization : ndarray
        Updated magnetization vector. The shapes of T1 and T2 are prefixed to the
        shape of the magnetization array if not already present.

    Examples
    --------
    # One line to simulate a relaxation process for 5000 time steps with 3000 parameter combos!
    import numpy as np
    from asl_bloch_sim import bloch

    dt = 0.001 # seconds
    duration = 5 # seconds
    T1 = np.linspace(0.3, 2.3, 20) # seconds
    T2 = np.linspace(0.05, 1.1, 30) # seconds
    mag = np.random.random((5, 3)) # last axis is the spatial axis
    mags = np.array([mag := bloch.relax(mag, T1, T2, dt) for _ in range(round(duration / dt))])
    mags.shape # (5000, 20, 30, 5, 3)

    """
    if axis < 0:
        axis += magnetization.ndim
    if (lmag := magnetization.shape[axis]) != len(M0):
        message = f"Length of M0: {len(M0)}, must be equal to the length of the spatial axis of the magnetization array: {lmag}"
        raise ValueError(message)
    try:
        T2, T1 = np.broadcast_arrays(T2, T1)
    except ValueError:
        T2, T1 = np.broadcast_arrays(T2, T1.reshape(T1.shape + (1,) * T2.ndim))
    transverse_relaxation_decay = 1 - dt / T2
    longitudinal_relaxation_decay = 1 - dt / T1
    longitudinal_relaxation_rise = dt / T1
    relaxation_decay = np.moveaxis([transverse_relaxation_decay,
                                    transverse_relaxation_decay,
                                    longitudinal_relaxation_decay], 0, -1)
    shape = tuple([-1 if ax == axis else 1 for ax in range(magnetization.ndim)])
    if match_shapes and extend_shapes and magnetization.shape[:T1.ndim] == T1.shape:
        outshape = T1.shape + shape[T1.ndim:]
    elif extend_shapes:
        outshape = T1.shape + shape
    else:
        outshape = list(magnetization.shape)
        outshape[axis] = -1
        relaxation_decay = np.swapaxes(relaxation_decay, -1, axis)[..., 0]
    rise = np.array(M0).reshape(shape) * longitudinal_relaxation_rise.reshape(outshape)
    return magnetization * relaxation_decay.reshape(outshape) + rise

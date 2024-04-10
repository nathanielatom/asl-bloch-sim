from asl_bloch_sim import xp, get_array_module
from asl_bloch_sim import progress_bar, progress_print

GAMMA_BAR = 42.5759e6 # Gyromagnetic ratio (Hz/T)
GAMMA = 2 * xp.pi * GAMMA_BAR # Gyromagnetic ratio (rads/T/s)

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

def construct_B_field(rf_am, G=0, position=0, *, off_resonance=0, B1_sensitivity=1,
                      rf_fm=0, axis=-1):
    """
    Construct the magnetic field vector from the RF and gradient waveforms.

    Parameters
    ----------
    rf_am : ndarray
        RF amplitude waveform in Tesla.
    G : ndarray, optional
        Gradient waveform in Tesla/m. Default is 0.
    position : ndarray, optional
        Spatial position vector in meters. Default is 0.
    off_resonance : float, optional
        Off-resonance frequency in Hz. Default is 0.
    B1_sensitivity : float or ndarray, optional
        B1 sensitivity factor. Default is 1.
    rf_fm : ndarray, optional
        RF frequency waveform in Hz. Default is 0.
    axis : int, optional
        Axis along the field array which represents the 2D or 3D spatial field vector.
        Also used for the dot product between gradient and position vectors. Default is -1.

    Returns
    -------
    ndarray
        Magnetic field vector in Tesla. Shape is extended from rf_am by G * position,
        off_resonance, B1_sensitivity, and coordinates for the spatial axis.

    """
    xp = get_array_module(rf_am)

    dBz = off_resonance / GAMMA_BAR # T
    rf_am = expand_dims_to(expand_dims_to(rf_am, dBz), B1_sensitivity)
    rf_am = (B1_sensitivity * rf_am).astype(xp.complex64) # T
    Bx, By = rf_am.real, rf_am.imag

    rf_fm = expand_dims_to(rf_fm, dBz)
    B1z = rf_fm / GAMMA_BAR # T
    Bz_pos = expand_dims_to(dot(G, position, axis=axis), dBz)
    Bz = expand_dims_to(Bz_pos + dBz + B1z, B1_sensitivity)

    B = xp.moveaxis(xp.asarray(xp.broadcast_arrays(Bx, By, Bz), dtype=xp.float32), 0, axis)

    from asl_bloch_sim import xp # use module level library (even if RF array was numpy)
    return xp.asarray(B)

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
    - If either `a` or `b` is a scalar, the dot product is computed as the element-wise multiplication of `a` and `b.
    - The dot product is computed using the Einstein summation convention.

    Examples
    --------
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
    xp = get_array_module(v, k, theta)
    if normalize:
        k = k / xp.linalg.norm(k, axis=axis, keepdims=True)
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)
    dot_kv = dot(k, v, axis=axis, keepdims=True)
    cross_kv = xp.cross(k, v, axis=axis)
    v_rot = v * cos_theta + cross_kv * sin_theta + k * dot_kv * (1 - cos_theta)
    return v_rot

def unit_field_and_angle(B_field, dt, *, tol=1e-14, axis=-1):
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
    xp = get_array_module(B_field)
    # Set length/magnitude of B
    B_length = xp.linalg.norm(B_field, axis=axis, keepdims=True)
    mask = B_length <= tol
    B_length[mask] = tol

    # Set rotation-axis (unit-vector) components
    b_field = B_field / B_length

    # Set rotation-angle [rads], note negative sign for LHR
    ang = -GAMMA * B_length * dt
    ang[mask] = 0
    return b_field, ang

def precess(magnetization, B_field, dt, *, axis=-1, **kwargs):
    magnetization = xp.asarray(magnetization, dtype=xp.float32)
    for dim in range(B_field.ndim - magnetization.ndim):
        magnetization = xp.expand_dims(magnetization, axis=0 if axis == -1 or axis == B_field.ndim - 1 else -1)
    return rodrigues_rotation(magnetization, *unit_field_and_angle(B_field, dt, axis=axis, **kwargs),
                              normalize=False, axis=axis)

def relax(magnetization, T1, T2, dt, *, M0=(0, 0, 1),
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
    xp = get_array_module(magnetization, T1, T2)
    if axis < 0:
        axis += magnetization.ndim
    if (lmag := magnetization.shape[axis]) != len(M0):
        message = f"Length of M0: {len(M0)}, must be equal to the length of the spatial axis of the magnetization array: {lmag}"
        raise ValueError(message)
    T1, T2 = xp.asarray(T1), xp.asarray(T2)
    try:
        T2, T1 = xp.broadcast_arrays(T2, T1)
    except ValueError:
        T2, T1 = xp.broadcast_arrays(T2, T1.reshape(T1.shape + (1,) * T2.ndim))
    transverse_relaxation_decay = 1 - dt / T2
    longitudinal_relaxation_decay = 1 - dt / T1
    longitudinal_relaxation_rise = dt / T1
    relaxation_decay = xp.moveaxis(xp.asarray([transverse_relaxation_decay,
                                               transverse_relaxation_decay,
                                               longitudinal_relaxation_decay]), 0, -1)
    shape = tuple([-1 if ax == axis else 1 for ax in range(magnetization.ndim)])
    if match_shapes and extend_shapes and magnetization.shape[:T1.ndim] == T1.shape:
        outshape = T1.shape + shape[T1.ndim:]
    elif extend_shapes:
        outshape = T1.shape + shape
    else:
        outshape = list(magnetization.shape)
        outshape[axis] = -1
        relaxation_decay = xp.swapaxes(relaxation_decay, -1, axis)[..., 0]
    M0 = xp.array(M0, dtype=xp.float32).reshape(shape)
    rise = M0 * longitudinal_relaxation_rise.reshape(outshape)
    return magnetization * relaxation_decay.reshape(outshape) + rise

def sim(B_field, T1, T2, duration, dt, *, init_mag=(0, 0, 1), **kwargs):
    mag = init_mag
    mags = xp.empty_like(B_field)
    for step in progress_bar(range(round(duration / dt))):
        mag = relax(precess(mag, B_field[step], dt, **kwargs), T1, T2, dt)
        mags[step] = mag
    return mags

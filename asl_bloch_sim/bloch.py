from asl_bloch_sim import xp, get_array_module
from asl_bloch_sim import progress_bar
from asl_bloch_sim import utils

GAMMA_BAR = 42.5759e6 # Gyromagnetic ratio (Hz/T)
GAMMA = 2 * xp.pi * GAMMA_BAR # Gyromagnetic ratio (rads/T/s)

def construct_B_field(rf_am, G=0, position=0, *, off_resonance=0, B1_sensitivity=1,
                      rf_fm=0, time_axis=0, space_axis=-1):
    """
    Construct the magnetic field vector from the RF and gradient waveforms.

    Parameters
    ----------
    rf_am : ndarray
        RF amplitude waveform in Tesla.
    G : ndarray, optional
        Gradient waveform in Tesla/m. The length of the
        time axis should match `rf_am`. Default is 0.
    position : ndarray, optional
        Spatial position vector in meters. Shape should match `G`.
        Default is 0.
    off_resonance : float, optional
        Off-resonance frequency in Hz. Default is 0.
    B1_sensitivity : float or ndarray, optional
        B1 sensitivity factor. Default is 1.
    rf_fm : ndarray, optional
        RF frequency waveform in Hz. Default is 0.
    time_axis : int, optional
        Axis along the input arrays which represents the time axis. Default is 0.
    space_axis : int, optional
        Axis along the field array which represents the 2D or 3D spatial field vector.
        Also used for the dot product between gradient and position vectors. Default is -1.

    Returns
    -------
    B_eff : ndarray
        Magnetic field vector in Tesla. Shape is extended from rf_am by :math:`G \\cdot position`,
        off_resonance, B1_sensitivity, and coordinates for the spatial axis. Explicitly, when space_axis=-1,
        the shape of the magnetic field vector is (time, ..., dB0, dB1, 3), where time is `len(rf_am)`,
        `dB0 = len(off_resonance)`, `dB1 = len(B1_sensitivity)`, `3` is the spatial axis (x, y, z), and
        `...` represents any extra dimensions in `rf_am` (for example to parametrize bandwidth), followed
        by any extra dimensions in `dot(G, position, space_axis)` (for example to parametrize spin motion).

    Notes
    -----
    As this function uses numpy-style broadcasting, the input arrays should have unique shapes, for example,
    parametrized dimensions like `len(off_resonance)` should NOT match `len(rf_am)`, the number of time steps.

    The magnetic field vector is computed as:

    .. math::

        B_x = \\Delta B_1 \\Re{RF_{AM}}
        B_y = \\Delta B_1 \\Im{RF_{AM}}
        B_z = G \\cdot r + \\frac{RF_{FM}}{\\gammabar} + \\frac{\\Delta f}{\\gammabar}

    where :math:`B_x`, :math:`B_y`, and :math:`B_z` are the magnetic field components,
    :math:`\\Delta B_1` is the unitless B1 sensitivity factor, :math:`RF_{AM}` is the RF amplitude modulation waveform in Tesla,
    :math:`G` is the gradient waveform in Tesla/m, :math:`r` is the spin's spatial position waveform in meters,
    :math:`RF_{FM}` is the RF frequency modulation waveform in Hz, :math:`\\gammabar` is the reduced gyromagnetic ratio in Hz/T,
    and :math:`\\Delta f` is the off-resonance frequency in Hz.

    """
    xp = get_array_module(rf_am)

    dBz = off_resonance / GAMMA_BAR # T
    rf_fm = utils.expand_dims_to(rf_fm, dBz)
    B1z = rf_fm / GAMMA_BAR # T
    Bz_pos = utils.expand_dims_to(utils.dot(G, position, axis=space_axis), dBz) # T
    Bz = utils.expand_dims_to(Bz_pos + dBz + B1z, B1_sensitivity).astype(xp.float32) # T

    rf_am = utils.expand_dims_to(rf_am, Bz, dimodifier=-int(utils.has_time_axis(Bz, rf_am, time_axis))) # -1 dim to dedupe time axis
    rf_am = (B1_sensitivity * rf_am).astype(xp.complex64) # T
    Bx, By = rf_am.real, rf_am.imag

    B = xp.moveaxis(xp.broadcast_arrays(Bx, By, Bz), 0, space_axis)

    from asl_bloch_sim import xp # use module level library (even if RF array was numpy)
    return xp.asarray(B)

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

    .. math::
        \\vec{b} = \\frac{\\vec{B}}{|B|}
        \\theta = -\\gamma |B| dt

    where :math:`|B|` is the magnitude of the field vector in Tesla,
    :math:`\\gamma` is the gyromagnetic ratio in rads/s/T, and :math:`dt` is the
    time step in seconds.

    """
    xp = get_array_module(B_field)
    # Calc the magnitude of B
    B_length = xp.linalg.norm(B_field, axis=axis, keepdims=True)
    mask = B_length <= tol
    B_length[mask] = tol

    # Set rotation-axis (unit-vector) components
    b_field = B_field / B_length

    # Set rotation-angle [rads], note negative sign for Left-Hand-Rule
    ang = -GAMMA * B_length * dt
    ang[mask] = 0
    return b_field, ang

def precess(magnetization, B_field, dt, *, axis=-1, **kwargs):
    """
    Simulate the precession of the magnetization vector under the influence of the magnetic field.

    This function simulates the precession of the magnetization vector according to the Bloch equations.
    The precession is computed using the Rodrigues' rotation formula.

    Parameters
    ----------
    magnetization : ndarray
        The initial magnetization vector.
    B_field : ndarray
        The magnetic field vector in Tesla.
    dt : float
        The time step in seconds.
    axis : int, optional
        The spatial axis along which precession takes place. Default is -1.
    **kwargs : dict
        Additional keyword arguments to pass to the `unit_field_and_angle` function.

    Returns
    -------
    ndarray
        The magnetization vector after precession.

    See Also
    --------
    unit_field_and_angle
    utils.rodrigues_rotation
    relax

    """
    magnetization = xp.asarray(magnetization, dtype=xp.float32)
    for dim in range(B_field.ndim - magnetization.ndim):
        magnetization = xp.expand_dims(magnetization, axis=0 if axis == -1 or axis == B_field.ndim - 1 else -1)
    return utils.rodrigues_rotation(magnetization, *unit_field_and_angle(B_field, dt, axis=axis, **kwargs),
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

    Raises
    ------
    ValueError: If the length of M0 is not equal to the length of the spatial axis
        of the magnetization array.

    Notes
    -----
    The relaxation is computed as:

    .. math::

        \\vec{M}_{\\text{relaxed}} = \\vec{M}_{\\text{stressed}} \\cdot \\left[1 - \\frac{\\Delta t}{T_2}, 1 - \\frac{\\Delta t}{T_2}, 1 - \\frac{\\Delta t}{T_1} \\right]^T + \\vec{M_0} \\left(\\frac{\\Delta t}{T_1} \\right)

    where :math:`\\vec{M}_{\\text{relaxed}}` is the relaxed magnetization vector,
    :math:`\\vec{M}_{\\text{stressed}}` is the input magnetization vector,
    :math:`\\vec{M_0}` is the initial magnetization vector,
    :math:`T_1` is the longitudinal relaxation time in seconds,
    :math:`T_2` is the transverse relaxation time in seconds,
    and :math:`\\Delta t` is the time step in seconds.

    Examples
    --------

    .. code-block:: python

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
    # TODO: construct B_field on the fly (at each time step to save memory)
    # and or precalc unit length b and theta to save processing time in loop
    # b, theta = bloch.unit_field_and_angle(B, dt)
    mag = init_mag
    mags = xp.empty_like(B_field)
    for step in progress_bar(range(round(duration / dt))):
        mag = relax(precess(mag, B_field[step], dt, **kwargs), T1, T2, dt)
        mags[step] = mag
    return mags

def inverted_magnetization(magnetization, time, T1, position, magindex=-1, time_axis=0, space_axis=-1, M0=1):
    """
    Calculate labelling efficiency from final longitudinal magnetization at the end of the
    simulation by correcting for the T1 decay experienced by the spin isochromat since
    crossing the labelling plane :cite:`Dai2008`.

    This provides a natural way to estimate the central tendency of the ringing/oscillatory magnetization
    signal at the time of inversion.

    Notes
    -----
    The longitudinal magnetization is corrected by:

    .. math::

        M_{inversion} = (M_{final} - M_0) \\exp\\left(\\frac{t_{final} - t_{inversion}}{T_1}\\right) + M_0

    where:
    - :math:`M_{inversion}` is the longitudinal magnetization resulting from inversion.
    - :math:`M_{final}` is the longitudinal magnetization at the final time point.
    - :math:`M_0` is the initial longitudinal magnetization.
    - :math:`t_{final}` and :math:`t_{inversion}` are final and inversion times, respectively.
    - :math:`T_1` is the longitudinal relaxation.

    References
    ----------
    W. Dai, D. Garcia, C. de Bazelaire, and D. C. Alsop, "Continuous flow-driven inversion for arterial spin
    labeling using pulsed radio frequency and gradient fields," Magnetic Resonance in Medicine, vol. 60,
    pp. 1488-1497, Nov. 2008.
    """
    xp = get_array_module(magnetization, position)

    if time_axis == 0 and space_axis == -1:
        final_mag = magnetization[magindex, ..., -1]
    else:
        final_mag = xp.take(xp.take(magnetization, magindex, axis=time_axis), -1, axis=space_axis)
    crossing = time[xp.abs(position).argmin(axis=time_axis)] # TODO should off reso crossing be modified?
    time_since = utils.expand_dims_to(time[magindex] - crossing, final_mag, collapse_matching=True) # s
    invert_mag = (final_mag - M0) * xp.exp(time_since / T1) + M0
    return invert_mag

def labelling_efficiency(long_mag_inverted, long_mag_control=1):
    """
    Compute the labeling efficiency from inverted and control longitudinal magnetizations.

    Parameters
    ----------
    long_mag_inverted : array_like or float
        Longitudinal magnetization measured in the inverted (label) condition. May be a scalar
        or an array; if an array, it must be broadcastable with long_mag_control.
    long_mag_control : array_like or float, optional
        Longitudinal magnetization measured in the control condition (default: 1). May be a scalar
        or an array; if an array, it must be broadcastable with long_mag_inverted.

    Returns
    -------
    float or ndarray
        The absolute labeling efficiency computed element-wise as
            |long_mag_control - long_mag_inverted| / |2 * long_mag_control|.
        The return has the same shape as the broadcasted inputs. For physically meaningful
        inputs, values typically lie in [0, 1].

    Notes
    -----
    - If long_mag_control is zero, the result is undefined (division by zero). When using NumPy,
      this will produce inf or nan values and may emit a runtime warning.
    - This function does not validate input types beyond relying on NumPy broadcasting semantics;
      pass numeric scalars or array-like objects.

    Examples
    --------
    >>> labelling_efficiency(0.2)
    0.4
    >>> labelling_efficiency(np.array([0.9, 0.8]))
    array([0.05, 0.1])
    """
    return xp.abs(long_mag_control - long_mag_inverted) / xp.abs(2 * long_mag_control)

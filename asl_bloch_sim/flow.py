import numpy as np
import scipy.signal as sig
from scipy.interpolate import CubicSpline

from asl_bloch_sim import utils

def integrate_trajectory(velocity_waveform, dt, position_offset=0, axis=-1):
    """
    Integrate a velocity waveform to obtain a position waveform, or sample bolus trajectory.

    The integration is performed using a cumulative sum.

    Parameters
    ----------
    velocity_waveform : ndarray
        Array of blood flow velocities in m/s.
    dt : float
        Time step between consecutive velocity measurements in seconds.
    initial_position : float, optional
        Initial position of the trajectory in m. Can be specified to offset the
        entire trajectory. Default is 0.
    axis : int, optional
        Axis along which to integrate, should correspond to the waveform time axis. Default is -1.

    Returns
    -------
    trajectory : ndarray
        Array of position in m, corresponding to the integrated trajectory.

    """
    return np.cumsum(velocity_waveform, axis=axis) * dt - position_offset

def constant(start=0, stop=2, num=1000, velocity=0.2, **kwargs):
    """
    Generate a constant velocity flow profile.

    Parameters
    ----------
    start : float
        The starting time in seconds. Default is 0.
    stop : float
        The ending time in seconds. Default is 2.
    num: int
        The number of time steps. Default is 1000.
    velocity : float
        The constant velocity in m/s. Default is 0.2.
    **kwargs : optional
        Additional keyword arguments to be passed to the
        `integrate_trajectory` function for calculating position.

    Returns
    -------
    time_steps : ndarray
        Array of time steps in seconds.
    velocity : ndarray
        Velocity waveform in m/s.
    position : ndarray
        Position waveform in m.

    """
    time_steps = np.linspace(start, stop, num)
    velocity = np.full(num, velocity)
    return time_steps, velocity, integrate_trajectory(velocity, dt=time_steps[1] - time_steps[0], **kwargs)

def half_sin(start=0, stop=2, num=1000, interbeat_interval=0.75,
             systolic_velocity=0.4, diastolic_velocity=0.01, phase=0, **kwargs):
    """
    Generate a blood flow velocity waveform using a half sine wave, where negative
    values are set to zero, then shifted with the diastolic velocity offset.

    A very approximate model of blood flow in the Aorta.

    Parameters
    ----------
    start : float, optional
        The starting time of the waveform in seconds. Default is 0.
    stop : float, optional
        The stopping time of the waveform in seconds. Default is 2.
    num : int, optional
        The number of time points to generate. Default is 1000.
    interbeat_interval : float, optional
        The interval between heartbeats (the pulse period) in seconds. Default is 0.75.
    systolic_velocity : float, optional
        Systolic velocity in m/s.
    diastolic_velocity : float, optional
        Diastolic velocity in m/s.
    phase : float, optional
        Phase shift in radians.
    **kwargs : optional
        Additional keyword arguments to be passed to the
        `integrate_trajectory` function for calculating position.

    Returns
    -------
    time_steps : ndarray
        Array of time steps in seconds.
    velocity_waveform : ndarray
        Velocity waveform in m/s.
    position : ndarray
        Position waveform in m.

    Examples
    --------

    .. bokeh-plot::

        import numpy as np
        from bokeh.plotting import figure, show
        from bokeh.io import output_notebook
        output_notebook()

        from asl_bloch_sim import flow

        systolic_velocities = np.linspace(0.01, 1, 50)[:, np.newaxis]
        time, velocity, position = flow.half_sin(systolic_velocity=systolic_velocities, phase=np.pi/8)

        plot = figure(title='Approximate Aortic Bloodflow Model', x_axis_label='Time (s)', y_axis_label='Bloodflow',
                      width=680, height=400)
        plot.line(time, velocity[40] * 100, legend_label='Velocity (cm/s)', line_color='blue')
        plot.line(time, position[40] * 100, legend_label='Position (cm)', line_color='purple')
        plot.legend.click_policy = 'hide'
        show(plot)

    """
    time_steps = np.linspace(start, stop, num)
    pulse = 1 / interbeat_interval
    pulse_train = 0.5*sig.square(2*np.pi*pulse*time_steps + phase, duty=0.5) + 0.5
    velocity_waveform = (systolic_velocity - diastolic_velocity) * pulse_train * np.sin(2*np.pi*pulse*time_steps + phase) + diastolic_velocity
    return time_steps, velocity_waveform, integrate_trajectory(velocity_waveform, dt=time_steps[1] - time_steps[0], **kwargs)

def exp_decay_train(start=0, stop=2, num=1000, interbeat_interval=0.917,
                    systolic_velocity=0.7, diastolic_velocity=0.08, decay=8, phase=0, **kwargs):
    """
    Generate a blood flow velocity waveform using an exponential decay train.

    This is a very crude yet smooth model of blood flow in the carotids
    based on empirical blood flow velocity from Doppler Ultrasound measurements.
    Estimated from a figure in an anesthesiology textbook that I unfortunately
    lost the reference for :( See `holdsworth_cca` for a more accurate CCA model.

    Parameters
    ----------
    start : float, optional
        The starting time of the waveform in seconds. Default is 0.
    stop : float, optional
        The stopping time of the waveform in seconds. Default is 2.
    num : int, optional
        The number of time points to generate. Default is 1000.
    interbeat_interval : float, optional
        The interval between heartbeats (the pulse period) in seconds. Default is 0.917.
    systolic_velocity : float, optional
        The peak systolic velocity in m/s. Default is 0.7.
    diastolic_velocity : float, optional
        The diastolic velocity in m/s. Default is 0.08.
    decay : float, optional
        The unitless decay rate of the exponential function. Default is 8.
    phase : float, optional
        The phase shift of the pulse train in radians. Default is 0.
    **kwargs : dict
        Additional keyword arguments to pass to the `integrate_trajectory` function.

    Returns
    -------
    time_steps : ndarray
        Array of time points in seconds.
    velocity_waveform : ndarray
        Array of blood flow velocities in m/s corresponding to the time points.
    position_waveform : ndarray
        Array of blood flow position for a sample bolus in m.

    Notes
    -----
    The exponential decay train is generated by convolving a pulse train (or dirac delta comb)
    with an exponential decay function.

    See Also
    --------
    holdsworth_cca

    Examples
    --------

    .. bokeh-plot::

        from bokeh.plotting import figure, show
        from bokeh.io import output_notebook
        output_notebook()

        from asl_bloch_sim import flow

        time, velocity, position = flow.exp_decay_train()

        plot = figure(title='Approximate Carotid Bloodflow Model', x_axis_label='Time (s)', y_axis_label='Bloodflow',
                      width=680, height=400)
        plot.line(time, velocity * 100, legend_label='Velocity (cm/s)', line_color='blue')
        plot.line(time, position * 100, legend_label='Position (cm)', line_color='purple')
        plot.legend.click_policy = 'hide'
        show(plot)

    """
    time_steps = np.linspace(start, stop, num)
    pulse = 1 / interbeat_interval
    phase += 3*np.pi/4 # approximate default phase shift to align with `holdsworth_cca`
    pulse_train = 0.5*sig.square(2*np.pi*pulse*time_steps + phase, duty=10/num) + 0.5 # delta comb function
    exp_decay = np.heaviside(time_steps, 1) * np.exp(-decay*pulse*time_steps) * decay*pulse*time_steps
    exp_train = sig.convolve(exp_decay, pulse_train, mode='full')[len(time_steps)//4:len(time_steps)+len(time_steps)//4]
    velocity_waveform = (systolic_velocity - diastolic_velocity) * exp_train / exp_train.max() + diastolic_velocity
    return time_steps, velocity_waveform, integrate_trajectory(velocity_waveform, dt=time_steps[1] - time_steps[0], **kwargs)

def holdsworth_cca(start=0, stop=2, num=1000, interbeat_interval=0.917,
                   systolic_velocity=0.76, diastolic_velocity=0.3, **kwargs):
    """
    Generate a blood flow velocity waveform for the common carotid artery (CCA)
    based on the model by Holdsworth et al. (1999) :cite:`Holdsworth1999`.

    Parameters
    ----------
    start : float, optional
        The starting time of the waveform in seconds. Default is 0.
    stop : float, optional
        The stopping time of the waveform in seconds. Default is 2.
    num : int, optional
        The number of time points to generate. Default is 1000.
    interbeat_interval : float, optional
        The interval between heartbeats (the pulse period) in seconds. Default is 0.917.
    systolic_velocity : float, optional
        The peak systolic velocity in m/s. Default is 0.76 following Zhao et al. (2016) :cite:`Zhao2016`.
    diastolic_velocity : float, optional
        The diastolic velocity in m/s. Default is 0.3 following Zhao et al. (2016) :cite:`Zhao2016`.
    **kwargs : dict
        Additional keyword arguments to pass to the `integrate_trajectory` function.

    Returns
    -------
    time_steps : ndarray
        Array of time points in seconds.
    velocity_waveform : ndarray
        Array of blood flow velocities in m/s corresponding to the time points.
    position_waveform : ndarray
        Array of blood flow position for a sample bolus in m.

    References
    ----------
    D. W. Holdsworth, C. J. D. Norley, R. Frayne, D. A. Steinman, and B. K. Rutt, "Characterization of
    common carotid artery blood-flow waveforms in normal human subjects," Physiological Measurement,
    vol. 20, p. 219-240, Jan. 1999. Available at: https://doi.org/10.1088/0967-3334/20/3/301

    L. Zhao, M. Vidorreta, S. Soman, J. A. Detre, and D. C. Alsop, "Improving the robustness of pseudo-
    continuous arterial spin labeling to off-resonance and pulsatile flow velocity," Magnetic Resonance in
    Medicine, vol. 78, p. 1342-1351, Oct. 2016. Available at: https://doi.org/10.1002/mrm.26513

    Notes
    -----
    The waveform is generated by interpolating key features of the cross-section's peak velocity time course
    in the CCA, averaged across subjects, taken from Table 2 in Holdsworth et al. (1999) :cite:`Holdsworth1999`.
    Additional average interpolated points are estimated and included in the model. A cubic spline
    `scipy.interpolate.CubicSpline` is fit to the data points, with a periodic boundary condition.

    Examples
    --------

    .. bokeh-plot::

        from bokeh.plotting import figure, show
        from bokeh.io import output_notebook
        output_notebook()

        from asl_bloch_sim import flow

        time, velocity, position = flow.holdsworth_cca(diastolic_velocity=0.12)

        plot = figure(title='Common Carotid Artery Bloodflow Model', x_axis_label='Time (s)', y_axis_label='Bloodflow',
                      width=680, height=400)
        plot.line(time, velocity * 100, legend_label='Velocity (cm/s)', line_color='blue')
        plot.line(time, position * 100, legend_label='Position (cm)', line_color='purple')
        plot.legend.click_policy = 'hide'
        show(plot)

    """
    features_time = [0, 0.055, 0.110, 0.116, 0.153, 0.116 + 0.103, 0.398, 0.917] # s
    features_velo = [25.9, 20.9, 47.7, 64.6, 108.2, 64.6, 19.4, 25.9] # cm/s
    estimated_interpolated_points_time = [0.25, 0.3, 0.35, 0.48, 0.6, 0.7, 0.8] # s
    estimated_interpolated_points_velo = [48, 46, 38, 40, 31, 28, 27.5] # cm/s
    knotx = np.concatenate((features_time, estimated_interpolated_points_time))
    naughty = np.concatenate((features_velo, estimated_interpolated_points_velo))
    naughty = naughty[np.argsort(knotx)]
    knotx.sort()
    knotx *= interbeat_interval / knotx[-1]
    naughty = naughty / 100 # cm/s to m/s
    spline_model = CubicSpline(knotx, naughty, bc_type='periodic')
    time_steps = np.linspace(start, stop, num)
    dt = time_steps[1] - time_steps[0]
    velocity_waveform = utils.binormalize(spline_model(time_steps), diastolic_velocity, systolic_velocity)
    return time_steps, velocity_waveform, integrate_trajectory(velocity_waveform, dt=dt, **kwargs)

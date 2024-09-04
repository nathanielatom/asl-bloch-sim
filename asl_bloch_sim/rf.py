import numpy as np
import scipy.signal as sig

from asl_bloch_sim import bloch

def sinc_pulse(flip_angle, duration, bandwidth, dt, phase_angle=0, window='hann'):
    """
    Generate a sinc pulse with a given flip angle and duration.

    Parameters
    ----------
    flip_angle : float
        Desired flip angle in degrees.
    duration : float
        Pulse duration in seconds.
    bandwidth : float
        Pulse bandwidth in Hz.
    dt : float
        Time step in seconds.
    phase_angle : float, optional
        Desired phase angle in degrees. Default is 0.
    window : str, optional
        Window function to apply to the sinc pulse. Default is Hann.

    Returns
    -------
    ndarray
        Sinc pulse waveform.

    Notes
    -----
    The flip angle of an RF pulse is given by the integral of the pulse B1 field:

    .. math::

            \\theta = \\gamma \\int B_1 dt

        where :math:`\\theta` is the flip angle, :math:`\\gamma` is the gyromagnetic
        ratio, and :math:`B_1` is the pulse amplitude. The sinc pulse is normalized
        to achieve the desired flip angle.

    """
    theta, alpha = np.deg2rad(phase_angle), np.deg2rad(flip_angle)
    time_bandwidth_product = duration * bandwidth
    x = np.linspace(-time_bandwidth_product / 2, time_bandwidth_product / 2,
                    round(duration / dt), endpoint=False)

    pulse = sig.get_window(window, len(x)) * np.sinc(x)
    B1 = np.exp(1j * theta) * alpha / (np.trapz(pulse, dx=dt) * bloch.GAMMA)

    return B1 * pulse # Tesla

def adiabaticity(pulse_am, pulse_fm, dt):
    """
    Compute the adiabaticity of an RF pulse.

    Parameters
    ----------
    pulse_am : ndarray
        Amplitude modulation waveform in Tesla.
    pulse_fm : ndarray
        Frequency modulation waveform in Hz, relative to the Larmor frequency.
    dt : float
        Time step in seconds.

    Returns
    -------
    ndarray
        Adiabaticity waveform.

    Notes
    -----
    The adiabaticity of an RF pulse is given by:

    .. math::

        K = \\frac{\\left | \\gamma B_{\\mathrm{effective}} \\right |}{\\left | \\dv{\\varphi}{t} \\right |} = \\frac{\\gamma\\sqrt{A^2(t) + \\left ( B_0 - \\frac{\\omega(t)}{\\gamma} \\right )^2}}{\\dv{}{t}\\left ( \\arctan(\\frac{A(t)}{B_0 - \\frac{\\omega(t)}{\\gamma}}) \\right )}

    where :math:`A(t)` and :math:`\\omega(t)` are the amplitude and frequency
    modulation waveforms, respectively, and :math:`B_0` is the static magnetic
    field. The adiabaticity is a measure of the ability of the pulse to drive
    the magnetization to follow the instantaneous effective magnetic field in
    the rotating frame. When the adiabaticity is much greater than 1, for all
    time, the pulse is considered adiabatic.

    """
    Bz_eff = pulse_fm / bloch.GAMMA_BAR
    return bloch.GAMMA * np.sqrt(pulse_am**2 + Bz_eff**2) / np.abs(np.gradient(np.arctan2(pulse_am, Bz_eff), dt))

def adiabatic_pulse(flip_angle, duration, bandwidth, stretch, dt, amplitude=1e-5, type='sech'):
    """
    Generate an adiabatic pulse with a given flip angle and duration.

    Parameters
    ----------
    flip_angle : float
        Desired flip angle in degrees. Not yet implemented.
    duration : float
        Pulse duration in seconds.
    bandwidth : float
        Pulse bandwidth in Hz.
    dt : float
        Time step in seconds.
    amplitude : float, optional
        Pulse amplitude in Tesla. Default is 10 µT. This may need to be adjusted along with the bandwidth.
    type : str, optional
        Type of adiabatic pulse. Default is hyperbolic secant.

    Returns
    -------
    pulse_am : ndarray
        Adiabatic pulse amplitude modulated waveform in Tesla.
    pulse_fm : ndarray
        Adiabatic pulse frequency modulated waveform in Hz.

    Notes
    -----
    The adiabatic pulse is generated by modulating the amplitude and frequency
    of the pulse waveform. The amplitude and frequency modulations are designed
    to achieve adiabaticity, i.e., to drive the magnetization to follow the
    instantaneous effective magnetic field in the rotating frame.

    """
    # TODO: implement flip angle for adiabadic pulse
    # pulse_am[-1] / pulse_fm[-1] = ratio
    ratio = np.tan(np.deg2rad(flip_angle)) / bloch.GAMMA_BAR

    time_bandwidth_product = duration * bandwidth
    x = np.linspace(-time_bandwidth_product / 2, time_bandwidth_product / 2,
                    round(duration / dt), endpoint=False) / stretch

    if type == 'sech':
        pulse_am = amplitude * np.cosh(x) ** -1
        pulse_fm = -bandwidth * np.tanh(x) / 2
    else:
        message = f'Unsupported adiabatic pulse type: {type}'
        raise ValueError(message)

    return pulse_am, pulse_fm

def extend(pulse, duration, dt, axis=0):
    """
    Extend the pulse waveform to a given duration.

    Parameters
    ----------
    pulse : ndarray
        Pulse waveform.
    duration : float
        Desired pulse duration in seconds.
    dt : float
        Time step in seconds.
    axis : int, optional
        Axis along which to extend the waveform. Default is 0.

    Returns
    -------
    ndarray
        Extended pulse waveform.

    Notes
    -----
    The pulse waveform is extended to the desired duration by appending zeros
    to the end of the waveform.

    """
    shap = list(np.shape(pulse))
    num_zero = round(duration / dt) - shap[axis]
    shap[axis] = num_zero
    zero = np.broadcast_to(bloch.expand_dims_to(np.zeros(num_zero), pulse, dimodifier=-1), shap)
    return np.append(pulse, zero, axis=axis)

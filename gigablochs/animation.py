import numpy as np
import scipy.signal as sig
from sympy.ntheory import divisors

from gigablochs import SHELL
if 'notebook' in SHELL:
    from IPython.display import Video, display

def round_divisor(numerator, denominator):
    """
    Round the ratio `numerator / denominator` to an integer by
    adjusting the denominator to the closest factor of the numerator.

    Parameters
    ----------
    numerator : int
        Numerator of the ratio, must be an integer.
    denominator : scalar
        Ratio denominator to be adjusted.

    Returns
    -------
    devisor : int
        The adjusted denominator, now an integer that divides numerator.

    """
    factors = np.array(divisors(numerator))
    return factors[abs(denominator - factors).argmin()]

def downsample(time_signal, new_time_increment, duration='~20', mode='filter', **kwargs):
    length = time_signal.shape[0]
    if approx := isinstance(duration, str) and duration.startswith('~'):
        duration = float(duration[1:])
    factor = length * new_time_increment / duration
    if approx:
        if factor >= 1:
            factor = round_divisor(length, factor)
        else:
            factor = 1 / round(1 / factor)
        duration = length * new_time_increment / factor
    # else use exact duration provided, must result in integer reduction factor and new shape
    time_steps = np.arange(0, duration, new_time_increment) # TODO: debug off-by-one error occasionally when upsampling
    if time_steps.size != (new_shape := length / factor) or (mode != 'fourier' and factor >= 1 and not factor.is_integer()):
        message = f'Desired {duration=}, {new_time_increment=} are incompatible with {length=} and result in downsampling {factor=} and {new_shape=}, please adjust to ensure integers'
        raise ValueError(message)
    if mode == 'fourier':
        kwargs.setdefault('window', 'rect')
        resampled = sig.resample(time_signal, length / factor, axis=0, **kwargs)
    elif mode == 'filter':
        up = 1 if factor >= 1 else round(1 / factor)
        down = factor if factor >= 1 else 1
        kwargs.setdefault('padtype', 'line')
        resampled = sig.resample_poly(time_signal, up, down, axis=0, **kwargs)
    elif mode == 'alias':
        if factor < 1:
            message = f'Aliasing mode is only possible for downsampling, got {factor=}'
            raise ValueError(message)
        if kwargs:
            message = f'got unexpected keyword arguments {kwargs=}'
            raise ValueError(message)
        resampled = time_signal[::factor]
    else:
        message = f'Unsupported {mode=}, must be in {{"fourier", "filter", "alias"}}'
        raise ValueError(message)
    return time_steps, resampled

def constant_speed(og_time_steps, new_time_steps):
    return (og_time_steps[-1] - og_time_steps[0]) / (new_time_steps[-1] - new_time_steps[0])

def rescale_Beff(Beff, arrow_length=1):
    Beff = Beff * 1e6 # µT
    Bmax = np.linalg.norm(Beff, axis=-1).max()
    return arrow_length * Beff / Bmax, Bmax

def bloch_sphere(magnetization, B_field=None, time_increments=0.1, speed=None,
                 traces=('magnetization', 'B_field_projection'),
                 engine='manim-cairo', prologue=True, preview=True, quality='low_quality',
                 progress_bar='display', max_files_cached=1000, max_width=85, **kwargs):
    """
    Animate magnetization and B-field vectors on a Bloch sphere.

    This function creates a 3D animation of magnetization evolving on the Bloch sphere,
    optionally showing the magnetic field and various traces.

    Parameters
    ----------
    magnetization : array_like
        Magnetization vectors to animate. Should be an array of shape (N, 3) where N is
        the number of time points and the last dimension contains the (x, y, z) components.
    B_field : array_like, optional
        Magnetic field vectors corresponding to the magnetization. Should have the same shape
        as magnetization. If provided, the field will be rescaled.
        Default is None, and no field vector is displayed.
    time_increments : float or array_like, optional
        Time increment(s) between consecutive frames. If scalar, the same increment is used
        for all time points. If array, should match the time dimension of magnetization.
        Default is 0.1.
    speed : float or array_like, optional
        Playback speed multiplier relative to real time, for informational display only.
        Default is None.
    traces : tuple of str, optional
        Trace types to display in the animation. Traces are either the projection of
        arrow tips onto the surface of the Bloch Sphere (ending in `_projection`) or the
        3D historical trajectory of the arrow tips in space, which can be useful to show
        when a vector has a smaller magnitude. Options include 'magnetization', 'B_field',
        and their '_projection's. Default is ('magnetization', 'B_field_projection').
    engine : str, optional
        Animation engine to use. Currently only 'manim-cairo' is supported.
        Default is 'manim-cairo'.
    prologue : bool, optional
        Whether to include a prologue sequence in the animation which displays the definition
        of magnetization and B-field vectors, along with the max B-field amplitude and animaton
        speed relative to real time. Default is True.
    preview : bool, optional
        Whether to preview the animation after rendering. When running in a Jupyter notebook
        environment, the animation is embedded and displayed as output. Default is True.
    quality : str, optional
        Rendering quality. Options include 'low_quality', 'medium_quality', 'high_quality',
        etc. Default is 'low_quality' since it takes significantly less time to render. Once you
        are happy with the animation, you can re-render at higher quality for hours to produce
        high definition production quality visuals.
    progress_bar : str, optional
        Progress bar display mode. Default is 'display'.
    max_files_cached : int, optional
        Maximum number of cached files for the rendering engine. Default is 1000 as there's many
        little files for each rotation time step.
    max_width : int, optional
        Maximum width percentage for video display in notebooks. Default is 85.
    **kwargs : dict, optional
        Additional keyword arguments passed to the manim configuration.

    Notes
    -----
    Raw Bloch simulation time history signals can be quite large, so consider downsampling the
    magnetization and B-field signals before passing to this function, but be wary to preserve
    the key frequency content of the signal and avoid downsampling too far - always inspect
    your data.

    See Also
    --------
    manim : Manim animation library.
    downsample : Resample time-domain signals via Fourier or filtering methods.
    gigablochs.backends.manim_cairo.BlochScene : Manim Cairo backend for Bloch sphere animations.

    """
    if np.isscalar(time_increments):
        time_increments = np.full_like(magnetization, time_increments)[..., 0]
    if engine == 'manim-cairo':
        from tqdm.auto import tqdm
        import manim.scene.scene
        manim.scene.scene.tqdm = tqdm

        from manim import config, tempconfig
        from gigablochs.backends.manim_cairo import BlochScene

        kwargs['quality'] = quality
        kwargs['progress_bar'] = progress_bar
        kwargs['max_files_cached'] = max_files_cached
        with tempconfig(kwargs):
            scene = BlochScene()
            scene.set_data(magnetization, rescale_Beff(B_field) if B_field is not None else (None, None),
                           time_increments, speed, traces, prologue)
            scene.render(preview and not 'notebook' in SHELL)

            if preview and 'notebook' in SHELL:
                vid = Video(config['output_file'], embed=True,
                            html_attributes=f'controls loop style="max-width: {max_width}%;"')
                display(vid)

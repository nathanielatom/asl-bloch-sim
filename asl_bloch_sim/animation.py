import numpy as np
import scipy.signal as sig
from sympy.ntheory import divisors

from asl_bloch_sim import SHELL
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

def speed(og_time_steps, new_time_steps):
    return (og_time_steps[-1] - og_time_steps[0]) / (new_time_steps[-1] - new_time_steps[0])
    # return np.gradient(og_time_steps).mean() / np.gradient(new_time_steps).mean()

def rescale_Beff(Beff, arrow_length=1):
    Beff = Beff * 1e6 # µT
    Bmax = np.linalg.norm(Beff, axis=-1).max()
    return arrow_length * Beff / Bmax, Bmax

def bloch_sphere(magnetization, B_field=None, time_increments=0.1, speed=None,
                 traces=('magnetization', 'B_field_projection'),
                 engine='manim-cairo', prologue=True, preview=True, quality='low_quality',
                 progress_bar='display', max_files_cached=1000, max_width=85, **kwargs):
    if np.isscalar(time_increments):
        time_increments = np.full_like(magnetization, time_increments)[..., 0]
    if engine == 'manim-cairo':
        from tqdm.auto import tqdm
        import manim.scene.scene
        manim.scene.scene.tqdm = tqdm

        from manim import config, tempconfig
        from asl_bloch_sim.backends.manim_cairo import BlochScene

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

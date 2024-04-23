import numpy as np

from asl_bloch_sim import SHELL
if 'notebook' in SHELL:
    from IPython.display import Video, display

def rescale_Beff(Beff, arrow_length=1):
    Beff = Beff * 1e6 # µT
    Bmax = np.linalg.norm(Beff, axis=-1).max()
    return arrow_length * Beff / Bmax, Bmax

def bloch_sphere(magnetization, B_field=None, time_increments=0.1, speed=None,
                 traces=('magnetization', 'B_field_projection'),
                 engine='manim-cairo', preview=True, quality='low_quality',
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
            scene.set_data(magnetization, rescale_Beff(B_field), time_increments, speed, traces)
            scene.render(preview and not 'notebook' in SHELL)

            if preview and 'notebook' in SHELL:
                vid = Video(config['output_file'], embed=True,
                            html_attributes=f'controls loop style="max-width: {max_width}%;"')
                display(vid)

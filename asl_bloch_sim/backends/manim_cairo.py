from manim import *
import numpy as np

def unit_vector(vect, fall_back=None):
    unit = normalize(vect)
    if np.all(unit == 0) and fall_back is not None:
        return fall_back
    return unit

def put_start_and_end_on(arrow, start, end, *, animate=False, small=1e-2):
    hawkeye = arrow.animate if animate else arrow
    if np.array_equal(start, end):
        if not getattr(arrow, 'shrunk', False):
            arrow.shrunk = True
            vect = unit_vector(arrow.get_end() - arrow.get_start())
            return hawkeye.put_start_and_end_on(start, vect * small + start)
        else:
            return hawkeye.set_opacity(0)
    else:
        arrow.shrunk = False
        arrow.set_opacity(1)
        arrow.cone.tip.set_opacity(0)
        arrow.cone.base_circle.set_opacity(0)
        return hawkeye.put_start_and_end_on(start, end)

class BlochScene(ThreeDScene):

    def set_data(self, magnetization, B_field, time_increments, speed=None,
                 traces=('magnetization', 'B_field_projection')):
        """
        Set the data for the Bloch sphere animation.

        Parameters
        ----------
        magnetization : ndarray, shape (time, n, 3)
            Magnetization vectors.
        B_field : tuple(ndarray, float)
            Tuple containing the normalized/rescaled effective B field and the maximum B field magnitude.
        time_increments : ndarray, shape (time,)
            Time increments for each animation step in seconds.
        speed : float, optional
            Displayed speed factor of the animation. Default is None.
        traces : sequence, optional
            Traced paths to display, 'magnetization' traces the value of the magnetization vector tip,
            'B_field' traces the value of the B_field vector tip, 'magnetization_projection' traces the projection
            of the magnetization vector onto the unit sphere, and 'B_field_projection' traces the projection
            of the B_field vector onto the unit sphere. Default is ('magnetization', 'B_field_projection').

        Example
        -------

        .. code-block:: python

            import numpy as np
            from asl_bloch_sim.backends.manim_cairo import BlochScene
            from asl_bloch_sim.animation import rescale_Beff
            sample_rate = 10 # Hz
            time = np.arange(0, 3, 1/sample_rate)
            time_increments = np.gradient(time)
            B_eff = np.repeat(np.array([[0, -1, 0]]), time.size, axis=0) * 1e-6 # T
            period = 10 # s
            mag = np.array([np.sin(2 * np.pi * time / period), np.zeros_like(time), np.cos(2 * np.pi * time / period)]).T
            scene = BlochScene()
            scene.set_data(mag, rescale_Beff(B_eff), time_increments)
            scene.render()

        """
        self.magnetization = magnetization
        self.B_field, self.B_max = B_field
        self.time_increments = time_increments
        self.speed = speed
        self.traces = traces

    def construct(self):
        start_point = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        ranges = [-1, 1, 1]
        length = 2
        axes = ThreeDAxes(x_range=ranges, y_range=ranges, z_range=ranges,
                          x_length=length, y_length=length, z_length=length,
                          axis_config=dict(include_tip=False))
        sphere = Sphere(radius=1, fill_opacity=0, stroke_opacity=0.25,  resolution=(32, 32))
        sphere_axes = Sphere(radius=1, fill_opacity=0, stroke_opacity=0.75,  resolution=(4, 2))

        x_label = axes.get_x_axis_label(Text("x", font_size=DEFAULT_FONT_SIZE / 2), direction=RIGHT).shift(DOWN * 0.1).set_z_index(1)
        y_label = axes.get_y_axis_label(Text("y", font_size=DEFAULT_FONT_SIZE / 2), direction=UP).shift(LEFT * 0.1).set_z_index(1)
        z_label = axes.get_z_axis_label(Text("z", font_size=DEFAULT_FONT_SIZE / 2), direction=OUT).shift(OUT * 0.1).set_z_index(1)

        size = 0.5
        thickness = 0.02 * size
        height = 0.3 * size
        base_radius = 0.08 * size

        if self.B_field is not None:
            field_arrow = Arrow3D(start=start_point, end=up, color='#8C50CC', # a better PURPLE,
                                  thickness=thickness, height=height, base_radius=base_radius).set_z_index(2)
            put_start_and_end_on(field_arrow, start_point, self.B_field[0])
            if 'B_field_projection' in self.traces:
                field_traced_projection = TracedPath(field_arrow.get_direction, stroke_color='#A875DE', stroke_opacity=0.8, stroke_width=3).set_z_index(2)
                self.add(field_traced_projection)
            if 'B_field' in self.traces:
                field_traced_path = TracedPath(field_arrow.get_end, stroke_color='#7432BB', stroke_opacity=0.6, stroke_width=3).set_z_index(2)
                self.add(field_traced_path)

        mag_arrow = Arrow3D(start=start_point, end=up, color=PINK,
                            thickness=thickness, height=height, base_radius=base_radius).set_z_index(2)
        put_start_and_end_on(mag_arrow, start_point, self.magnetization[0])
        if 'magnetization_projection' in self.traces:
            mag_traced_projection = TracedPath(mag_arrow.get_direction, stroke_color='#F67795', stroke_opacity=0.3, stroke_width=4).set_z_index(2)
            self.add(mag_traced_projection)
        if 'magnetization' in self.traces:
            mag_traced_path = TracedPath(mag_arrow.get_end, stroke_color='#F25277', stroke_opacity=0.8, stroke_width=4).set_z_index(2)
            self.add(mag_traced_path)

        try:
            MathTex('\\gamma')
            latex = True
        except RuntimeError as error:
            message = 'latex failed but did not produce a log file. Check your LaTeX installation.'
            if message in str(error):
                latex = False
            else:
                raise error

        if latex:
            latex_template = TexTemplate()
            latex_template.add_to_preamble("\\usepackage[a]{esvect}")

        mag_text = MathTex("\\vv{M}(t)", font_size=DEFAULT_FONT_SIZE * 1.5,
                           color=PINK, tex_template=latex_template).to_corner(UL) if latex else Text("M(t)", font_size=DEFAULT_FONT_SIZE * 1.5, color=PINK).to_corner(UL)
        self.add_fixed_in_frame_mobjects(mag_text)
        self.remove(mag_text)

        if self.B_field is not None:
            field_text = MathTex("\\vv{B}_{\\textrm{eff}}(t)", font_size=DEFAULT_FONT_SIZE * 1.5,
                                color=PURPLE, tex_template=latex_template).to_corner(UR) if latex else Text("B(t)", font_size=DEFAULT_FONT_SIZE * 1.5, color=PURPLE).to_corner(UR)
            self.add_fixed_in_frame_mobjects(field_text)
            self.remove(field_text)

            scale_text = MathTex("\\left|\\vv{B}_{\\textrm{eff}}(t)\\right|_{\\textrm{max}} = %s \\mu T" % f'{self.B_max:.3g}', font_size=DEFAULT_FONT_SIZE * 0.75,
                                color=PURPLE, tex_template=latex_template).to_corner(DR) if latex else Text("max|B(t)| = %s µT" % f'{self.B_max:.3g}', font_size=DEFAULT_FONT_SIZE * 0.75, color=PURPLE).to_corner(DR)
            self.add_fixed_in_frame_mobjects(scale_text)
            self.remove(scale_text)

        if self.speed is not None:
            speed_text = Text(f"Speed: {self.speed:.3g}x", font_size=DEFAULT_FONT_SIZE * 0.75, color=WHITE).to_corner(DL)
            self.add_fixed_in_frame_mobjects(speed_text)
            self.remove(speed_text)

        self.set_camera_orientation(zoom=3, phi=70 * DEGREES, theta=-70 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.01)

        self.play(FadeIn(axes), FadeIn(sphere), FadeIn(sphere_axes), run_time=2)
        self.play(Write(x_label), Write(y_label), Write(z_label), run_time=1)
        self.wait(1)
        self.play(Create(mag_arrow), run_time=0.5)
        self.play(Write(mag_text), run_time=1)
        if self.B_field is not None:
            self.wait(1)
            self.play(Create(field_arrow), run_time=0.5)
            self.play(Write(field_text), run_time=1)
            self.wait(1)
            self.play(Write(scale_text), run_time=1)
        self.wait(1)
        if self.speed is not None:
            self.play(Write(speed_text), run_time=1)
        self.wait(2)

        # Animate the movement of the mag_arrow
        for dt, field, mag in zip(self.time_increments, self.B_field, self.magnetization):
            mag_arrow.direction = unit_vector(mag - start_point, fall_back=mag_arrow.direction)
            updates = [put_start_and_end_on(mag_arrow, start_point, mag, animate=True)]
            if self.B_field is not None:
                field_arrow.direction = unit_vector(field - start_point, fall_back=field_arrow.direction)
                updates.append(put_start_and_end_on(field_arrow, start_point, field, animate=True))
            self.play(*updates, rate_func=linear, run_time=dt)

        self.wait()

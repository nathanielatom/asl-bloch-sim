from manimlib import *
import numpy as np

class ArrowAnimation(ThreeDScene):
    def construct(self):
        ranges = [-1, 1, 1]
        length = 2
        axes = ThreeDAxes(x_range=ranges, y_range=ranges, z_range=ranges,
                          depth=length,
                          axis_config=dict(include_tip=False))
        sphere = Sphere(radius=1, opacity=0, resolution=(32, 32)) # , stroke_opacity=0.25
        sphere_axes = Sphere(radius=1, opacity=0, resolution=(4, 2)) # , stroke_opacity=0.75

        # Define the start and end points
        start_point = np.array([0, 0, 0])
        framerate = 10 # Hz
        time = np.arange(0, 3, 1/framerate)
        period = 10 # s
        end_points = np.array([np.sin(2 * np.pi * time / period), np.zeros_like(time), np.cos(2 * np.pi * time / period)]).T

        # Create an mag_arrow
        mag_arrow = Arrow(start=start_point, end=end_points[0], color=PINK, opacity=1)
        # mag_arrow.cone.show_base = True
        field_arrow = Arrow(start=start_point, end=[0, -1, 0], color=PURPLE, opacity=1)
        # field_arrow.cone.show_base = True

        # latex_template = TexTemplate()
        # latex_template.add_to_preamble("\\usepackage[a]{esvect}")
        #, tex_template=latex_template
        DEFAULT_FONT_SIZE = 48
        mag_text = Text("M(t)", font_size=DEFAULT_FONT_SIZE * 2,
                           color=PINK, is_fixed_in_frame=True).to_corner(UL)
        field_text = Text("B_eff(t)", font_size=DEFAULT_FONT_SIZE * 2,
                             color=PURPLE, is_fixed_in_frame=True).to_corner(UR)

        # Create a traced path
        traced_path = TracedPath(mag_arrow.get_end, stroke_color=ORANGE, stroke_opacity=0.8, stroke_width=5)

        x_label = axes.get_x_axis_label("x", direction=RIGHT).shift(DOWN * 0.1)
        y_label = axes.get_y_axis_label("y", direction=UP).shift(LEFT * 0.1) #.rotate(-90 * DEGREES).shift(UP * 0.1) # .shift(DOWN * 0.5)
        z_label = axes.get_axis_label("z", axes.get_z_axis(), edge=OUT, direction=OUT).shift(OUT * 0.1) # axes.get_z_axis_label("z", direction=OUT).shift(OUT * 0.1)

        self.camera.frame.set_euler_angles(phi=70, theta=30, units=DEGREES) # (90, 30, -90-70, units=DEGREES)
        # self.camera.frame.set_focal_distance(self.camera.frame.get_focal_distance() * 3) # zoom in
        # self.set_camera_orientation(zoom=3, phi=70 * DEGREES, theta=-70 * DEGREES)
        # self.begin_ambient_camera_rotation(rate=0.01)
        self.camera.frame.add_updater(lambda m, dt: m.increment_theta(0.01 * dt))
        # add xtimes speed on LR

        self.play(FadeIn(axes), FadeIn(sphere), FadeIn(sphere_axes), run_time=2)
        self.play(Write(x_label), Write(y_label), Write(z_label), run_time=1)
        self.wait(1)
        self.play(Write(mag_arrow), Write(mag_text), run_time=1)
        self.wait(1)
        self.play(Write(field_arrow), Write(field_text), run_time=1)
        self.wait(2)
        self.play(FadeOut(mag_text), FadeOut(field_text), run_time=0.5)
        self.wait(1)
        self.add(traced_path)

        # Animate the movement of the mag_arrow
        for end_point in end_points:
            self.play(
                mag_arrow.animate.put_start_and_end_on(start_point, end_point),
                rate_func=linear,
                run_time=1/framerate
            )

        self.wait()
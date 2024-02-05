import manim as mn

class Rotation3DExample(mn.ThreeDScene):
    def construct(self):
        axes = mn.ThreeDAxes()
        sphere = mn.Sphere(radius=2, fill_opacity=0, stroke_opacity=0.25,  resolution=(16, 16))
        sphere_axes = mn.Sphere(radius=2, fill_opacity=0, stroke_opacity=0.5,  resolution=(2, 2))
        magnetization = mn.Arrow3D([0, 0, 0], [0, 0, 2], resolution=8, color=mn.PINK)
        b_field = mn.Arrow3D([0, 0, 0], [0, 1, 0], resolution=8, color=mn.PURPLE)

        x_label = axes.get_x_axis_label(mn.Text("x"))
        y_label = axes.get_y_axis_label(mn.Text("y")).shift(mn.UP * 1.8)
        z_label = axes.get_z_axis_label(mn.Text("z")).shift(mn.OUT)

        self.begin_ambient_camera_rotation(rate=0.1)
        self.set_camera_orientation(zoom=1, phi=75 * mn.DEGREES, theta=30 * mn.DEGREES)

        self.play(mn.FadeIn(axes), mn.FadeIn(x_label), mn.FadeIn(y_label), mn.FadeIn(z_label),
                  mn.FadeIn(sphere), mn.FadeIn(sphere_axes))
        self.wait(1)
        self.play(mn.Write(magnetization), run_time=0.5)
        self.wait(0.5)
        self.play(mn.Write(b_field), run_time=0.5)
        self.wait(2)

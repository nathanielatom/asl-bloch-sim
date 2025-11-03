from manim import config
from manim import *
from manim.opengl import *
from manim.mobject.opengl.opengl_geometry import OpenGLArrow

import numpy as np

class Rotation3DExample(ThreeDScene):

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     # data = np.load('examples/mags.npz')
    #     # self.mags, self.B, self.dt, self.dfz = data['mags'], data['B'], data['dt'], data['dfz']


    def construct(self):
        self.framerate = 5 # Hz
        self.speed = 1 # 0.125
        self.dt = 1 / self.framerate
        time = np.arange(0, 8, self.dt)
        period = 10 # s
        self.mags = np.array([np.zeros_like(time), np.sin(2 * np.pi * time / period), np.cos(2 * np.pi * time / period)]).T
        self.B = None # np.zeros_like(self.mags) + 0.01

        ranges = [-2, 2, 1]
        length = 4
        self.zero = np.array([0, 0, 0])
        self.axes = ThreeDAxes(x_range=ranges, y_range=ranges, z_range=ranges,
                                  x_length=length, y_length=length, z_length=length)
        self.sphere = Sphere(radius=1, fill_opacity=0, stroke_opacity=0.25,  resolution=(32, 32))
        self.sphere_axes = Sphere(radius=1, fill_opacity=0, stroke_opacity=0.75,  resolution=(4, 2))
        self.magnetization = OpenGLArrow(self.zero, self.mags[0], resolution=8, color=PINK)
        # self.b_field = OpenGLArrow(self.zero, self.B[0], resolution=8, color=PURPLE)

        # self.traced_path = TracedPath(self.magnetization.get_end, stroke_color=PINK, stroke_opacity=0.8, stroke_width=5)
        x_label = self.axes.get_x_axis_label(Text("x"))
        y_label = self.axes.get_y_axis_label(Text("y")).rotate(-90 * DEGREES).shift(DOWN * 0.5)
        z_label = self.axes.get_z_axis_label(Text("z")).shift(OUT * 0.2)

        self.begin_ambient_camera_rotation(rate=0.05)
        self.set_camera_orientation(phi=75 * DEGREES, theta=(-90 -80) * DEGREES) # zoom=1.75,
        # shift camera out up-on-z-axis
        # label M and B and keep speedx on screen

        self.play(FadeIn(self.axes), FadeIn(x_label), FadeIn(y_label), FadeIn(z_label),
                  FadeIn(self.sphere), FadeIn(self.sphere_axes))
        self.wait(1)
        self.play(Write(self.magnetization), run_time=0.5)
        # self.wait(0.5)
        # self.play(Write(self.b_field), run_time=0.5)
        # self.add(self.traced_path)
        self.wait(2)
        # normalize B vector
        #  / np.linalg.norm(self.B, axis=-1, keepdims=True)
        self.animate_sim(self.mags, self.B, self.dt, framerate=self.framerate, speed=self.speed)

    def animate_time_step(self, magnetization_vector, b_field_vector, framerate=5):
        self.play(self.magnetization.animate.put_start_and_end_on(self.zero, magnetization_vector),
                #   self.b_field.animate.put_start_and_end_on(self.zero, b_field_vector),
                  rate_func=linear,
                  run_time=1 / framerate)

    @staticmethod
    def downsample(arr, dt_old, dt_new, speed=1):
        """
        Naively downsample the input array. May cause aliasing.

        Parameters
        ----------
        arr : array_like
            Input array to be downsampled. Axis 0 is time.
        dt_old : float
            Time step of the original array.
        dt_new : float
            Time step of the downsampled array.
        speed : float, optional
            Speed factor for downsampling. Default is 1.

        Returns
        -------
        array_like
            Downsampled array.

        Raises
        ------
        ValueError
            If the downsampling factor is not an integer.

        Notes
        -----
        The downsampling factor is calculated as `speed * dt_new / dt_old`. To
        avoid aliasing, the frequency content of the original signal should be
        less than half of `1 / dt_new`.

        Examples
        --------
        >>> arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> dt_old = 0.1
        >>> dt_new = 0.2
        >>> speed = 2
        >>> downsample(arr, dt_old, dt_new, speed)
        [1, 5, 9]

        """
        factor = speed * dt_new / dt_old
        if factor.is_integer():
            return arr[::round(factor)]
        message = f"Downsampling factor {factor} is not an integer. "
        raise ValueError(message)

    def animate_sim(self, magnetization_vectors, b_field_vectors, dt, framerate=5, speed=1):
        mags = self.downsample(magnetization_vectors, dt, 1 / framerate, speed)
        # Bs = self.downsample(b_field_vectors, dt, 1 / framerate, speed)
        # mags, Bs = mags[:, 0], Bs[:, 0]
        # for mag, B in zip(mags, Bs):
        for mag in mags:
            B = None
            self.animate_time_step(mag, B, framerate)

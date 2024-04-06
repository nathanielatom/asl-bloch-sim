from manim import *
import numpy as np

class ArrowAnimation(Scene):
    def construct(self):
        # Define the start and end points
        start_point = np.array([0, 0, 0])
        framerate = 10 # Hz
        time = np.arange(0, 8, 1/framerate)
        period = 10 # s
        end_points = np.array([2 * np.sin(2 * np.pi * time / period), 2 * np.cos(2 * np.pi * time / period), np.zeros_like(time)]).T

        # Create an arrow
        arrow = Arrow(start=start_point, end=end_points[0])
        # end_points *= np.linalg.norm(arrow.get_vector()) / np.linalg.norm(arrow.end)
        # adjust end points so that the tip of the arrow is at the end point
        arrow.put_start_and_end_on(start_point, end_points[0])

        # Create a traced path
        traced_path = TracedPath(arrow.get_end, stroke_color=ORANGE, stroke_width=2)

        # Add the arrow and traced path to the scene
        self.add(arrow, traced_path)

        # Animate the movement of the arrow
        for end_point in end_points:
            self.play(
                arrow.animate.put_start_and_end_on(start_point, end_point),
                rate_func=linear,
                run_time=1/framerate
            )

        self.wait()
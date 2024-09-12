.. asl_bloch_sim documentation master file, created by
   sphinx-quickstart on Wed Sep 11 15:20:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

asl_bloch_sim documentation
===========================

Arterial Spin Labelling (ASL) is a magnetic resonance imaging (MRI) technique that uses the
magnetization of blood as an endogenous contrast agent. The Bloch equations describe the
time-evolution of the magnetization in a magnetic field. This package provides a simple
simulation of the Bloch equations for ASL, including the effects of radiofrequency (RF) pulses,
gradients, and pulsatile blood flow.

Features
--------

- Simulate the full time-evolution of the magnetization following an arbitrary pulse sequence
- Uses numpy-style broadcasting to effectively run up to millions of Bloch simulations in parallel
- GPU acceleration using CuPy, with CPU fallback, and fully device-agnostic code
- Visualise the effect of whole parameter grids on magnetization results using Bokeh
- Animations of the magnetization dynamics in 3D on the Bloch Sphere using Manim

Example
-------

Pulsed ASL (PASL) uses an adiabatic pulse to label a slab of blood in one go. Here's an
animation of an example adiabatic pulse:

.. video:: _static/BlochScene_PASL.mp4
    :width: 700
    :loop:

|

Installation
------------

See dependencies in the `pyproject.toml` file for required packages. To install asl_bloch_sim
from github run:

.. code-block:: bash

    pip install git+https://github.com/nathanielatom/asl-bloch-sim

API Reference
-------------

.. autosummary::
    :toctree: _generated/

    asl_bloch_sim.utils.expand_dims_to
    asl_bloch_sim.utils.dot
    asl_bloch_sim.utils.rodrigues_rotation
    asl_bloch_sim.bloch.construct_B_field
    asl_bloch_sim.bloch.unit_field_and_angle
    asl_bloch_sim.bloch.precess
    asl_bloch_sim.bloch.relax
    asl_bloch_sim.rf.sinc_pulse
    asl_bloch_sim.rf.adiabatic_pulse
    asl_bloch_sim.rf.adiabaticity
    asl_bloch_sim.rf.extend
    asl_bloch_sim.flow.integrate_trajectory
    asl_bloch_sim.flow.constant
    asl_bloch_sim.flow.half_sin
    asl_bloch_sim.flow.exp_decay_train
    asl_bloch_sim.flow.holdsworth_cca

.. toctree::
   :maxdepth: 2
   :caption: Contents:

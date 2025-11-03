.. gigablochs documentation master file, created by
   sphinx-quickstart on Wed Sep 11 15:20:52 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gigablochs documentation
========================

GigaBlochs is a flexible Bloch simulation framework enabling large parameter space
investigations in Magnetic Resonance Imaging (MRI). The Bloch equations describe the
time-evolution of magnetization in a magnetic field resulting from radiofrequency (RF) pulses,
gradients waveforms, off-resonance effects, and motion of hydrogen nuclei (or "spins") through
these fields, as in the case of pulsatile blood flow in arteries.

The capability to handle arbitrary, multi-dimensional spin properties which evolve at every
timestep, combined with tools for flow modelling, allows for realistic simulations of
complex sequences, such as Arterial Spin Labelling (ASL). ASL is a technique that uses the
magnetization of blood as an endogenous contrast agent for quantification of perfusion.

Features
--------

- Simulate the full time-evolution of the magnetization following an arbitrary pulse sequence
- Avoids the hard pulse approximation, constructing all B-field components from RF and gradient waveforms along with off-resonance effects
- Uses numpy-style broadcasting to effectively run up to millions of Bloch simulations in parallel
- Sophisticated blood flow modelling tools to incorportate friction and pulsatile flow effects when relevant
- GPU acceleration using CuPy, with CPU fallback, for seamless and fully device-agnostic simulation code
- Visualise the effect on magnetization of whole parameter grids using Bokeh
- Animations of the magnetization dynamics in 3D on the Bloch Sphere using Manim

ASL Features
------------

- Simulate PASL sequences or background suppression using adiabatic pulses
- Simulate CASL with constant concurrent RF and gradients
- Simulate PCASL with a train of RF sinc pulses and gradients
- Use arbitrary flow trajectories to simulate the effect of pulsatile blood flow
- Use the rigid tube model to simulate viscosity and vessel wall friction across arteries

Example
-------

Pulsed ASL (PASL) uses an adiabatic pulse to label a slab of blood in one go. Here's an
animation on the Bloch sphere of an adiabatic inversion pulse in a reference frame
rotating at the instantaneous frequency of the pulse (which only coincides with the
Larmor frequency when passing through the transverse plane):

.. video:: _static/BlochScene_PASL.mp4
    :width: 700
    :loop:

|

.. code-block:: python

    import numpy as np
    from gigablochs import animation
    animation.bloch_sphere(downsampled_magnetization, downsampled_b_field, np.gradient(time_steps))

See `examples/Adiabatic Inversion Animation.ipynb <https://github.com/nathanielatom/gigablochs/blob/main/examples/Adiabatic%20Inversion%20Animation.ipynb>`__
for the full simulation code.

Installation
------------

See dependencies in the `pyproject.toml` file for required and optional packages. To install gigablochs
from github run:

.. code-block:: bash

    pip install git+https://github.com/nathanielatom/gigablochs

API Reference
-------------

.. autosummary::
    :toctree: _generated/

    gigablochs.animation.bloch_sphere
    gigablochs.backends.manim_cairo.BlochScene
    gigablochs.utils.expand_dims_to
    gigablochs.utils.dot
    gigablochs.utils.rodrigues_rotation
    gigablochs.bloch.construct_B_field
    gigablochs.bloch.unit_field_and_angle
    gigablochs.bloch.precess
    gigablochs.bloch.relax
    gigablochs.bloch.inverted_magnetization
    gigablochs.bloch.labelling_efficiency
    gigablochs.rf.sinc_pulse
    gigablochs.rf.adiabatic_pulse
    gigablochs.rf.adiabaticity
    gigablochs.rf.extend
    gigablochs.flow.integrate_trajectory
    gigablochs.flow.constant
    gigablochs.flow.half_sin
    gigablochs.flow.exp_decay_train
    gigablochs.flow.holdsworth_cca

.. toctree::
   :maxdepth: 2
   :caption: Contents:

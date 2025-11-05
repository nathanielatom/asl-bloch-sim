Adiabatic Pulse Example
=======================

.. py-config::

    splashscreen:
        autoclose: true
    packages:
    - numpy
    - scipy
    - bokeh

.. py-repl::
    :output: repl_output

    print("hallo world")
    import numpy as np
    import scipy.signal as sig

    t = np.linspace(0, 1, 100)
    pulse = sig.windows.tukey(100, alpha=0.5)
    print(pulse)

.. raw:: html

    <div id="repl_output"></div>

.. py-terminal::

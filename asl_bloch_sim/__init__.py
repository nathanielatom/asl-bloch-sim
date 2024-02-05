import os as _os
import warnings as _warnings

__version__ = "0.0.1"

def get_array_module(*args):
    import numpy as np
    return np

_desired_module = _os.environ.get("ARRAY_MODULE", 'numpy')
if _desired_module == 'cupy':
    try:
        import cupy as xp
        from cupy import get_array_module
        gpu = True
    except ModuleNotFoundError as error:
        import numpy as xp
        gpu = False
elif _desired_module == 'numpy':
    import numpy as xp
    gpu = False
else:
    _message = f"Invalid module: {_desired_module}. Use 'numpy' or 'cupy'."
    _warnings.warn(_message)
    import numpy as xp
    gpu = False

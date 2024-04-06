import os as _os
import warnings as _warnings

__version__ = "0.0.1"

def get_array_module(*args):
    import numpy as np
    return np

_desired_module = _os.environ.get("ARRAY_MODULE", 'cupy')
if _desired_module == 'cupy':
    try:
        import cupy as xp
        try:
            xp.zeros(1, dtype=xp.float32)
            from cupy import get_array_module
            gpu = True
        except xp.cuda.runtime.CUDARuntimeError as cuda_error:
            import numpy as xp
            gpu = False
    except ModuleNotFoundError as import_error:
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

asnumpy = xp.asnumpy if gpu else lambda arr, *args, **kwargs: arr

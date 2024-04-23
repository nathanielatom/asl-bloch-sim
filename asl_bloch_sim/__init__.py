import os as _os
import warnings as _warnings

from tqdm.auto import tqdm

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

def _get_shell_type():
    """
    Returns the current shell type, that is one of `pq.utils.SHELL_TYPES`.
    """
    try:
        shell_types = {"<class 'google.colab._shell.Shell'>": 'colaboratory notebook',
                       "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>": 'jupyter notebook',
                       "<class 'IPython.terminal.interactiveshell.TerminalInteractiveShell'>": 'ipython'}
        return shell_types[str(type(get_ipython()))]
    except (NameError, KeyError):
        pass

    return 'python'

SHELL = _get_shell_type()
SHELL_TYPES = {'python', 'ipython', 'jupyter notebook', 'colaboratory notebook'}
# convenience and user code readability
progress_bar = tqdm
progress_print = tqdm.write

import inspect
from typing import Any, Dict, Tuple
import fire

def get_function_args() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Get the args and kwargs of the currently executing function.
    Returns a tuple of (args, kwargs)
    """
    # Get the frame of the calling function
    frame = inspect.currentframe()
    try:
        # Get the frame of the parent (the actual function we want to inspect)
        frame = frame.f_back
        # Get the arguments as a mapping of parameter names to values
        args_info = inspect.getargvalues(frame)
        
        # Extract positional arguments
        # args = args_info.locals[arg] for arg in args_info.args)
        
        # Extract keyword arguments (excluding positional args)
        kwargs = {
            key: value 
            for key, value in args_info.locals.items()
            if key in args_info.args and not key.startswith('__')
        }
        
        return kwargs
    finally:
        # Always delete the frame reference to avoid reference cycles
        del frame

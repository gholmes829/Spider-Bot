"""

"""

from functools import wraps
from time import time

def timed(f):
    @wraps(f)
    def g(*args, **kwargs):
        start = time()
        out = f(*args, **kwargs)
        elapsed = time() - start
        return *out, elapsed
    return g

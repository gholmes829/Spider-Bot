"""

"""

import numpy as np
from functools import wraps

def validate_shape(f):
    @wraps(f)
    def g(*args, **kwargs):
        """
        Makes assertions to catch errors with shape and normalization.
        """
        assert len(args) == 2, f'Recieved unexpected positional arguments {args}'
        agent, observation = args
        assert all([-1.01 <= e <= 1.01 for e in observation]), f'Expected argument normalized in [-1, 1], recieved {observation}'
        input_error_msg = f'Expected input shape of {agent.input_shape}, got {len(observation)}'
        assert len(observation) == agent.input_shape, input_error_msg
        output = f(*args, **kwargs)
        output_error_msg = f'Expected output shape of {agent.output_shape}, got {len(output)}'
        assert len(output) == agent.output_shape, output_error_msg
        return output
    return g

class Agent:
    def __init__(self, model, input_shape, output_shape) -> None:
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = model
        self.i = 0
    
    @validate_shape
    def predict(self, observation):    
        # make ai prediction based on observation
        self.i += 1
        return self.model.activate(observation)
        # return 0.5 * np.cos(np.full(12, self.i / 25))
"""

"""

import numpy as np
from icecream import ic
from functools import wraps

def validate_shape(f = None, *, terminate_on_bound_error = True):
    def parameterized_validate_shape(g):
        @wraps(g)
        def h(*args, **kwargs):
            """
            Makes assertions to catch errors with shape and normalization.
            """
            assert len(args) == 2, f'Recieved unexpected positional arguments {args}'  # should just recieve agent (self) and obersvation
            agent, observation = args
            try:
                assert all([-1.01 <= e <= 1.01 for e in observation]), f'Expected argument normalized in [-1, 1], recieved {observation}'
            except AssertionError as ae:
                if terminate_on_bound_error:
                    raise ae
                else:
                    print(f'Failed to normalize inputs, ignoring', flush=True)
            input_error_msg = f'Expected input shape of {agent.input_shape}, got {len(observation)}'
            assert len(observation) == agent.input_shape, input_error_msg
            output = g(*args, **kwargs)
            output_error_msg = f'Expected output shape of {agent.output_shape}, got {len(output)}'
            assert len(output) == agent.output_shape, output_error_msg
            return output
        return h
    if f:
        return parameterized_validate_shape(f)
    else:
        return parameterized_validate_shape

class Agent:
    def __init__(self, model, input_shape, output_shape, id=None) -> None:
        self.model = model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.fitness = None
        self.id = id
    
    @validate_shape(terminate_on_bound_error = False)
    def predict(self, observation):
        # make ai prediction based on observation
        return self.model.activate(observation)
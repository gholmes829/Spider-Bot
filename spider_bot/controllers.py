import numpy as np
from numba import jit
from genetics import Member

class FFNN (Member):
	"""
	Feed forward neural network without backpropogation.
	Public Methods
	--------------
	feedForward(inputs) -> np.ndarray:
		Feeds inputs through neural net to obtain output.
	"""
	def __init__(self, layerSizes: list, id: int = None, activation: str = "sigmoid", weights: list = None, biases: list = None) -> None:
        super(Member, self).__init__(id)

		"""
		Initializes.
		Parameters
		----------
		layerSizes: list
			Layer architecture
        id: int
            unique base-32 indentifier
		activation: str, default="sigmoid"
			String denoting activation function to use
		weights: list, optional
			List of arrays of weights for each layer, randomized if not passed in
		biases: list, optional
			List of arrays of biases for each layer, randomized if not passed in
		"""

		activations = {
			"sigmoid": FFNN.sigmoid,
			"reLu": FFNN.reLu,
			"softmax": FFNN.softmax
		}
		self.layerSizes = layerSizes
		weightShapes = [(i, j) for i, j in zip(layerSizes[1:], layerSizes[:-1])]
		self.weights = [np.random.randn(*s) for s in weightShapes] if weights is None else weights
		self.biases = [np.random.standard_normal(s) for s in layerSizes[1:]] if biases is None else biases
		self.activation = [activations[activation] 

	def predict(self, inputs: np.ndarray) -> np.ndarray:
		"""
		Feeds inputs through neural net to obtain output.
		Parameters
		----------
		inputs: np.ndarray
			Inputs to neural network
		Returns
		-------
		np.ndarray: input after fed through neural network
		"""
		for w, b in zip(self.weights, self.biases):
			inputs = self.activation(inputs @ w.T + b)
		return inputs

	@staticmethod
	@jit(nopython=True)
	def sigmoid(x: float) -> float:
		"""Sigmoid."""
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def reLu(x: float) -> float:
		"""Rectified linear unit."""
		return np.maximum(0, x)

	@staticmethod
	@jit(nopython=True)
	def softmax(v: np.ndarray) -> np.ndarray:
		"""Softmax probability output."""
		e = np.exp(v)
		return e / e.sum()


from numba import jit
import numpy as np

class FFNN:
	"""
	Feed forward neural network without backpropogation.
	Public Methods
	--------------
	feedForward(inputs) -> np.ndarray:
		Feeds inputs through neural net to obtain output.
	"""
	def __init__(self, layerSizes: list, activation: str = "sigmoid", weights: list = None, biases: list = None) -> None:
		"""
		Initializes.
		Parameters
		----------
		layerSizes: list
			Layer architecture
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

def mutate(network: object) -> object:
    """
    Provides mutated clone of network.
    Mutates by assigning random value to a given number of 
    weights, biases, or activations

    """
    mutant = deepcopy(network)
    target = choice(("weights", "biases"))

    if target == "weights":
        randLayer = randint(0, len(network["weights"]) - 1)
        randNeuron = randint(0, len(network["weights"][randLayer]) - 1)
        randWeight = randint(0, len(network["weights"][randLayer][randNeuron]) - 1)
        network["weights"][randLayer][randNeuron][randWeight] = np.random.randn()
    else:
        randLayer = randint(0, len(network["biases"]) - 1)
        randBias = randint(0, len(network["biases"][randLayer]) - 1)
        network["biases"][randLayer][randBias] = np.random.randn()
            
    return mutant
    
def _crossover(self, parent1: object, parent2: object) -> object:
    """
    Cross over traits from parents for new member.
    Decides to cross over parents' weights or biases. Selects a random neuron to crossover.

    """
    child1, child2 = deepcopy(parent1), deepcopy(parent2)
    target = choice(("weights", "biases"))

    randLayer = randint(0, len(child[target]) - 1)
    randTargetElement = randint(0, len(child1[target][randLayer]) - 1)
    temp = deepcopy(child1)
    child1[target][randLayer][randTargetElement] = child2[target][randLayer][randTargetElement]
    child2[target][randLayer][randTargetElement] = temp[target][randLayer][randTargetElement]

    # fitness1, fitness2 = self.task(child1)["fitness"], self.task(child2)["fitness"]
    best, parentId = (child1, parent2.id) if fitness1 > fitness2 else (child2, parent2.id)
    self._updateMemberId(best)
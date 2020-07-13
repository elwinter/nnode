"""
SLFFNN - Base class for single-layer feed-forward neural networks

This module provides the base functionality for all single-layer feed-forward
neural network objects used in the nnode software.

Example:
    Create an empty SLFFNN object.
        net = SLFNN()

Attributes:
    None

Methods:
    None

Todo:
    None
"""

from neuralnetwork import NeuralNetwork

class SLFFNN(NeuralNetwork):
    """Base class for all single-layer feed-forward neural network objects"""

    def __init__(self):
        """Initialize the neural network object."""
        super().__init__()


if __name__ == '__main__':
    net = SLFFNN()
    print(net)

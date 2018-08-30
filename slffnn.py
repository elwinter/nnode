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
    * Expand base functionality.
"""

from neuralnetwork import NeuralNetwork

class SLFFNN(NeuralNetwork):
    """Base class for all single-layer feed-forward neural network objects"""
    pass

if __name__ == '__main__':
    net = SLFFNN()
    print(net)

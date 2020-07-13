"""
NeuralNetwork - Base class for neural networks

This module provides the base functionality for all neural network objects
used in the nnode software.

Example:
    Create an empty NeuralNetwork object.
        net = NeuralNetwork()

Attributes:
    None

Methods:
    train() - Stub for training methods for subclasses
    run() - Stub for run methods for subclasses

Todo:
    None
"""

class NeuralNetwork:
    """Base class for all neural network objects"""

    def __init__(self):
        """Initialize the neural network object."""
        pass

    def train(self):
        """Train the neural network."""
        pass

    def run(self):
        """Run the neural network."""
        pass

if __name__ == '__main__':
    net = NeuralNetwork()
    print(net)

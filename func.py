"""Create a base abstract class for mathematical functions."""

from ABC import abstractmethod
from scipy.special import expit  # pylint: disable=no-name-in-module


class MathFunction():
    """Class to describe a math function and some of its uses."""

    @abstractmethod
    def eval(self):
        """Find function value at point."""
        pass

    @abstractmethod
    def prime(self):
        """One variable derivative."""


def QuadraticCost(MathFunction):
    """Implement C(y) = 1/2 (y - a)^2."""

    @staticmethod
    def eval()



def Sigmoid(MathFunction):
    """Implement sigmoid(z)."""

    @staticmethod
    def eval(z):
        return expit(z)

    @staticmethod
    def prime(z):
        return expit(z) * (1 - expit(z))
"""Create a base abstract class for mathematical functions."""

from abc import abstractmethod
from scipy.special import expit  # pylint: disable=no-name-in-module
import numpy as np


class MathFunction():
    """Class to describe a math function and some of its uses."""

    @abstractmethod
    def eval(z, y, a):
        """Find function value at point."""

    @abstractmethod
    def prime(z, y, a):
        """One variable derivative and leave it in vectorized form."""


class QuadraticCost(MathFunction):
    """Implement C(y) = 1/2 (y - a)^2."""

    @staticmethod
    def eval(z, y, a):
        """Eval for quadratic cost."""
        return 0.5 * (np.linalg.norm(y - a))

    @staticmethod
    def prime(z, y, a):
        """Compute derivative vector for quadratic cost."""
        sig_prime = expit(z) * (1 - expit(z))
        return (y - a) * sig_prime


class CrossEntropy(MathFunction):
    """Implement C(y) = "-aln(y) - (1 - a)ln(1 - y)."""

    @staticmethod
    def eval(z, y, a):
        """Eval for cross entropy cost."""
        return -np.sum(np.nan_to_num(a * np.log(y) - (1 - a) * np.log(1 - y)))

    def prime(z, y, a):
        """Compute derivative vector for cross entropy cost."""
        return y - a

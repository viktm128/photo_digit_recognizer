"""Create a base abstract class for mathematical functions."""

from ABC import abstractmethod
from scipy.special import expit  # pylint: disable=no-name-in-module


class MathFunction():
    """Class to describe a math function and some of its uses."""

    @abstractmethod
    def eval(z, y, a):
        """Find function value at point."""
        pass

    @abstractmethod
    def prime(z, y, a):
        """One variable derivative and leave it in vectorized form."""
        pass


class QuadraticCost(MathFunction):
    """Implement C(y) = 1/2 (y - a)^2."""

    @staticmethod
    def eval(z, y, a):
        return 0.5 * (np.linalg.norm(y - a))

    @staticmethod
    def prime(z, y, a):
        return (y - a)


class CrossEntropy(MathFunction):
    """Implement C(y) = "-aln(y) - (1 - a)ln(1 - y)"""

    @staticmethod
    def eval(z, y, a):
        return -np.sum(np.nan_to_num(a * np.log(y) - (1 - a) * np.log(1 - y)))

    def prime(z, y, a):
        sig_prime = expit(z) * (1 - expit(z))
        return (y - a) / sig_prime
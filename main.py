"""Write command line interface for network."""

import click
import network


@click.command()
@click.option('--eta', type=float, default=3,
              help='Parameter to control step size during SDG.', show_default=True)
@click.option('--batch_size', default=10,
              help='Parameter to control sample size in each SDG step.', show_default=True)
@click.option('--epoch_max', default=10,
              help='Parameter to control how long to make the network learn.', show_default=True)
@click.option('--test', default=True, help='Output between each epoch.', show_default=True)
@click.option('--cross_entropy', default=True, help='Cross entropy cost vs. quadratic cost.', show_default=True)
@click.option('--lmbda', default=0.1, help='A number geq 0 inclusive to control amount of regularization.', show_default=True)
@click.argument('layers', type=int, nargs=-1)
def main(eta, batch_size, epoch_max, test, cross_entropy, lmbda, layers):
    """Call upon the Network class with certain hyper parameters.

    The class will always learn based on the parameters provided and then
    its output can be tweaked.

    LAYERS is a tuple of the sizes of each activation layer. For example,
    calling main.py 784 30 10 will create a structure of 784 input neurons,
    10 output neurons, and 30 hidden neurons in a single layer.
    """
    h_params = {
        "eta": eta,
        "batch_size": batch_size,
        "epoch_max": epoch_max,
        "cross_entropy": cross_entropy,
        "lmbda": lmbda
    }
    n = network.Network(h_params, test, layers)
    n.learn()


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

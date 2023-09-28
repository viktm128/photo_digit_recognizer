"""Write command line interface for network."""

import click
import network


@click.command()
@click.option('--eta', default=3,
              help='Parameter to control step size during SDG.')
@click.option('--batch_size', default=10,
              help='Parameter to control sample size in each SDG step.')
@click.option('--epoch_max', default=10,
              help='Parameter to control how long to make the network learn.')
@click.option('--test', default=True, help='Output between each epoch.')
@click.argument('layers', type=int, nargs=-1)
def main(eta, batch_size, epoch_max, test, layers):
    """Call upon the Network class with certain hyper parameters.

    The class will always learn based on the parameters provided and then
    its output can be tweaked.

    LAYERS is a tuple of the sizes of each activation layer. For example,
    calling main.py 784 30 10 will create a structure of 784 input neurons,
    10 output neurons, and 30 hidden neurons in a single layer.
    """
    n = network.Network(eta, batch_size, test, layers)
    n.learn(epoch_max)


if __name__ == '__main__':
    main()
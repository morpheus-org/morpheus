import argparse

def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """

    parser = argparse.ArgumentParser(
        description='Data processing, analysis and plotting for morpheus benchmarks')

    parser.add_argument('--filename', '-f', type=str, required=True,
                        help='CSV file to read the data from (absolute path to file)')

    parser.add_argument('--outdir', '-o', type=str, required=True,
                        help='Absolute path of the output directory to write the plot files in.')

    args = parser.parse_args()

    return args
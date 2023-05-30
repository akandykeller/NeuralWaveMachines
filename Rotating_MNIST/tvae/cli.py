import sys
import argparse
from tvae.experiments import (
    tcornn_2d, tcornn_1d
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'tvae.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()

if __name__ == "__main__":
    main()

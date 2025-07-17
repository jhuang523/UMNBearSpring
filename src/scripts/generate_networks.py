import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../src"))) #use this to be able to import local packages

from utils.utils import print_verbose
from utils.conduits import * 

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n', type = int, default = 10, help = "number of networks to generate")
    parser.add_argument('--settings_file', type = str, default = 'conduits.yaml', help = 'yaml file for network settings')
    parser.add_argument('--output_dir', type = str, default = 'conduit_networks', help = 'output dir for generated networks')
    parser.add_argument('--fname', type = str, default = 'network', help = 'file name pattern for exported networks')
    args = parser.parse_args()
    n = args.n
    settings_file=args.settings_file
    output_dir = args.output_dir
    fname = args.fname
    generate_n_networks(n, settings_file, output_dir, fname)

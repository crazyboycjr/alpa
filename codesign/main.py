#!/usr/bin/env python3
import time
import argparse

from config import Config
from train import train_with_alpa

'''
example: ./main.py --dry-run
example: NCCL_DEBUG=info ./main.py --config codesign/models/config.toml
'''

def add_args(parser):
    parser.add_argument(
        '-D',
        '--dry-run',
        action='store_true',
        help='Do not run. Only print what settings was chosen to run.')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='codesign/models/config.toml',
        help='The path to the configuration.')


def main(args):
    # load config
    config = Config.load(args.config)

    # enumerate and traverse all the cases
    for cluster_spec in config.generate_cluster_specs():
        for model_spec, training_spec in config.generate_models(cluster_spec):
            for parallel_method in config.generate_parallel_methods(cluster_spec):
                time.sleep(1)
                train_with_alpa(args, cluster_spec, model_spec, training_spec,
                                parallel_method)


if __name__ == '__main__':
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    # parse args
    parser = argparse.ArgumentParser(
        description="allreduce completion time calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_args(parser)
    args = parser.parse_args()
    main(args)
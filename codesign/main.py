#!/usr/bin/env python3
import argparse

from config import Config
from train import train_with_alpa
from db import DB

'''
example: ./main.py --dry-run
example: NCCL_DEBUG=info ./main.py --config codesign/models/config.toml
example: CODESIGN_LOG_LEVEL=info NCCL_DEBUG=info codesign/main.py
example: CODESIGN_LOG_LEVEL=info NCCL_DEBUG=info codesign/main.py --search-model
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
    parser.add_argument(
        '-d',
        '--db',
        type=str,
        default='codesign/results.db',
        help='The path to the sqlite3 database file. All the results will be saved there.')
    parser.add_argument(
        '--search-model',
        action='store_true',
        help='Whether to search the model. If not, use the models specified in the config.toml.')
    parser.add_argument(
        '--manual-job-timeout',
        type=int,
        default='600',
        help='The timeout threshold in seconds for jobs using manual stage. (Default: 10min)')
    parser.add_argument(
        '--retry-failed',
        action='store_true',
        help='Whether to retry failed jobs in the results database.')


def main(args):
    if args.search_model:
        import search
        model_specs = search.search_model()
        print(model_specs)
        return
    
    # load config
    config = Config.load(args.config)
    # connect to db
    db = DB(args.db)

    # enumerate and traverse all the cases
    for cluster_spec in config.generate_cluster_specs():
        for model_spec, training_spec in config.generate_models(cluster_spec):
            for parallel_spec in config.generate_parallel_specs(cluster_spec):
                train_with_alpa(args, db, cluster_spec, model_spec, training_spec,
                                parallel_spec)


if __name__ == '__main__':
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    # parse args
    parser = argparse.ArgumentParser(
        description="allreduce completion time calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_args(parser)
    args = parser.parse_args()
    main(args)
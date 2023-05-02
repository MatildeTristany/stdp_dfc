"""
This is a script to run a model with a certain hyperparameter 
fixed configuration.
"""


import importlib
import argparse
import sys
import numpy as np
import main


def _override_cmd_arg(config):
    sys.argv = [sys.argv[0]]
    for k, v in config.items():
        if isinstance(v, bool):
            cmd = '--%s' % k if v else ''
        else:
            cmd = '--%s=%s' % (k, str(v))
        if not cmd == '':
            sys.argv.append(cmd)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_module', type=str,
                        default='configs.configs_mnist.mnist_ndi',
                        help='The name of the module containing the config.')
    args = parser.parse_args()
    config_module = importlib.import_module(args.config_module)
    _override_cmd_arg(config_module.config)
    summary = main.run()
    return summary

if __name__ == '__main__':
    run()
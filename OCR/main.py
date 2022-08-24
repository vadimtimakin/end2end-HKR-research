import sys

from utils import set_seed
from train_functions import run, run_eval
from config import config

if __name__ == '__main__':
    set_seed(config.seed)
    if len(sys.argv) > 1:
        run_eval(config, sys.argv[1])
    else:
        run(config=config)
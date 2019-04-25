import tensorflow as tf
from models import Transformer,create_masks,LableSmoothingLoss
from utils import AttrDict,init_logger
import yaml,argparse

logger = init_logger('train.log')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/hparams.yaml')
    parser.add_argument('-load_model', type=str, default=None)
    # parser.add_argument('-fp16_allreduce', action='store_true', default=False,
    #                     help='use fp16 compression during allreduce')
    # parser.add_argument('-batches_per_allreduce', type=int, default=1,
    #                     help='number of batches processed locally before '
    #                          'executing allreduce across workers; it multiplies '
    #                          'total batch size.')
    parser.add_argument('-num_wokers', type=int, default=0,
                        help='how many subprocesses to use for data loading. '
                             '0 means that the data will be loaded in the main process')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile))



if __name__=='__main__':
    main()
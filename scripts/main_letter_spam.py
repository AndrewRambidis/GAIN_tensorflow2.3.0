'''
Main function for UCI letter and spam datasets
'''

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from tools.data_loader import data_loader
from tools.gain import gain
from tools.utils import rmse_loss

def main(args):
    '''
    Main function for UCI letter and spam datasets

    Args: 
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch_size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

    Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
    '''

    data_name = args.data_name
    miss_rate = args.miss_rate

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # load data and introduce missingless
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

    # convert to float32
    ori_data_x = ori_data_x.astype('float32')
    miss_data_x = miss_data_x.astype('float32')
    data_m = data_m.astype('float32')
    
    # impute missing data
    imputed_data_x = gain(miss_data_x, gain_parameters)

    # report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    
    print()
    print('RMSE Performance: ' + str(np.round(rmse, 4)))
    print(imputed_data_x)

    return imputed_data_x, rmse

if __name__ == '__main__':

    # inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data_name', 
            choices=['letter', 'spam'],
            default='spam',
            type=str
            )
    parser.add_argument(
            '--miss_rate',
            help='missing data probability',
            default=0.2,
            type=float
            )
    parser.add_argument(
            '--batch_size',
            help='number of samples in mini-batch',
            default=128,
            type=int
            )
    parser.add_argument(
            '--hint_rate',
            help='hint probability',
            default=0.9,
            type=float
            )
    parser.add_argument(
            '--alpha',
            help='hyperparameter',
            default=100,
            type=float
            )
    parser.add_argument(
            '--iterations',
            help='number of training iterations',
            default=10000,
            type=int
            )

    args = parser.parse_args()

    # calls main function
    imputed_data, rmse = main(args)


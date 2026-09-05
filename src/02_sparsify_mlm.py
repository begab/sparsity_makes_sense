import os

import scipy
import numpy as np

from utils.utils import *

from tqdm.auto import tqdm

import argparse
import logging
import logging.config
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates sparse contextualized representation.')
    parser.add_argument('--in_files', nargs='+', required=True) # it is assumed that the first file is used to determine matrix D

    parser.add_argument('--top-k', type=int)
    parser.add_argument('--nucleus', type=float)

    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--not-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    logging.info(args)

    for in_file in args.in_files:

        out_probs = np.load(in_file)

        to_keep_mask = np.zeros_like(out_probs, dtype=bool)
        if args.top_k:
            assert args.top_k <= out_probs.shape[1]

            sorted_indices = np.argsort(out_probs, axis=1)[:, -args.top_k:]
            for row, sis in enumerate(sorted_indices):
                to_keep_mask[row, sis] = True
            out_file_name = f'{in_file}_top-k_{args.top_k}'
          
        elif args.nucleus:
            assert 0 <= args.nucleus <= 1.0
            sorted_indices = np.argsort(out_probs, axis=1)[:, ::-1]
            cumsum = np.cumsum(np.sort(out_probs, axis=1)[:,::-1], axis=1)
            for row in range(out_probs.shape[0]):
                index = np.searchsorted(cumsum[row], args.nucleus)
                to_keep_mask[row, sorted_indices[row, 0:index+1]] = True
            out_file_name = f'{in_file}_nucleus_{args.nucleus}'
        else:
            mean = np.mean(out_probs, axis=0)
            median = np.median(out_probs, axis=0)
            to_keep_mask = out_probs > mean
            out_file_name = f'{in_file}_above_mean'

        out_probs[~to_keep_mask] = 0
        alphas = scipy.sparse.csr_matrix(out_probs)
        print(f"{out_file_name} {alphas.shape} nnz={100*alphas.nnz / np.prod(out_probs.shape):.2f}%")
        scipy.sparse.save_npz(out_file_name, alphas)


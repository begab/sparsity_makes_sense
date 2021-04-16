import os

import scipy
import spams
import numpy as np

from utils.utils import *

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

    parser.add_argument('--K', type=int, default=2000)
    parser.add_argument('--lda', type=float, default=0.05)
    parser.add_argument('--predefined_dictionary_file', type=str, default=None)

    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--not-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()

    logging.info(args)

    params = {'K': args.K, 'lambda1': args.lda, 'numThreads': 8, 'iter': 1000, 'batchsize': 400, 'posAlpha': True, 'verbose': False}
    lasso_params = {x:params[x] for x in ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose'] if x in params}
    lasso_params['pos'] = True
    dict_file = args.predefined_dictionary_file # when follows random:x pattern, the dictionary is randomly generated using seed x
    D = None

    for i, in_file in enumerate(args.in_files):
        print(in_file)
        assert os.path.exists(in_file)

        embeddings = np.load(in_file)
        if args.normalize:
            embeddings = row_normalize(embeddings)
        embeddings = embeddings.T
        if not np.isfortran(embeddings):
            embeddings = np.asfortranarray(embeddings)

        if dict_file is None:
            dict_file = '{}_norm{}_K{}_lda{}'.format(in_file, args.normalize, args.K, args.lda)

            if not os.path.exists('{}.npy'.format(dict_file)):
                logging.info("Dictionary learning for embeddings of shape: {}".format(embeddings.shape))
                D = spams.trainDL(embeddings, **params)
                np.save(dict_file, D)
                logging.info("Dictionary learning completed...")
            else:
                logging.info('Dictionary file already exists')
                continue
        elif D is None and dict_file.startswith('random:'):
            seed = int(dict_file.split(':')[1])
            dict_file = '{}_norm{}_K{}_lda{}_rnd{}'.format(in_file, args.normalize, args.K, args.lda, seed)
            np.random.seed(seed)
            D = col_normalize(embeddings @ np.random.randn(embeddings.shape[1], args.K)).astype(embeddings.dtype)
            dd = D.T @ D
            ddd = [dd[i,j] for i in range(dd.shape[0]) for j in range(dd.shape[1]) if i>j]
            logging.info((np.mean(ddd), np.std(ddd), np.min(ddd), np.max(ddd)))
            np.save(dict_file, D)
            logging.info('Random dictionary generated using {}.'.format(dict_file))

        alphas_file = '{}_{}_norm{}_K{}_lda{}'.format(dict_file, os.path.basename(in_file), args.normalize, args.K, args.lda)
        logging.info((dict_file, alphas_file))

        D = np.load('{}.npy'.format(dict_file))
        if not np.isfortran(D):
            D=np.asfortranarray(D)
        logging.info((D.dtype, embeddings.dtype, D.shape, embeddings.shape))

        alphas = spams.lasso(embeddings, D=D, **lasso_params)
        scipy.sparse.save_npz(alphas_file, alphas.T)
        logging.info((alphas_file, alphas.nnz, alphas.shape, alphas.nnz/alphas.shape[1]))

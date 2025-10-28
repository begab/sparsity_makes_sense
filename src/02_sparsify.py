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

    parser.add_argument('--K', type=int, default=3000)
    parser.add_argument('--lda', type=float, default=0.05)
    parser.add_argument('--predefined_dictionary_file', type=str, default=None)

    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--not-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--use-spams', dest='spams', action='store_true')
    parser.add_argument('--not-spams', dest='spams', action='store_false')
    parser.set_defaults(spams=True)

    parser.add_argument('--dl_iter', type=int, default=1000)
    
    args = parser.parse_args()

    logging.info(args)

    params = {'K': args.K, 'lambda1': args.lda, 'numThreads': 8, 'iter': args.dl_iter, 'batchsize': 512, 'posAlpha': True, 'verbose': False}
    lasso_params = {x:params[x] for x in ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose'] if x in params}
    lasso_params['pos'] = True
    dict_file = args.predefined_dictionary_file # when follows random:x pattern, the dictionary is randomly generated using seed x
    D = None
    if args.spams:
        import spams
    else:
        import torch
        from utils.sparser import FISTA, quadraticBasisUpdate

    for i, in_file in enumerate(args.in_files):
        assert os.path.exists(in_file)

        embeddings = np.load(in_file)
        if args.normalize:
            embeddings = row_normalize(embeddings)
        embeddings = embeddings.T
        if not np.isfortran(embeddings):
            embeddings = np.asfortranarray(embeddings)

        if dict_file is None:
            dict_file = f'{in_file.replace(".npy", "")}_norm{args.normalize}_K{args.K}_lda{args.lda}{"" if args.spams else "_torch"}_{params["iter"]}it'

            if not os.path.exists(f'{dict_file}.npy'):
                logging.info(f"Dictionary learning for embeddings of shape: {embeddings.shape}")
                if args.spams:
                    D = spams.trainDL(embeddings, **params)
                else:
                    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
                    dict_size = [embeddings.shape[0], args.K] # number of dimensions x bases
                    torch.manual_seed(42)
                    D = torch.randn(dict_size).to(device)
                    D = D.div_(D.norm(2, 0))
                    ACT_HISTORY_LEN = 300
                    HessianDiag = torch.zeros(args.K).to(device)
                    progress = tqdm(range(params['iter'] * embeddings.shape[1]))
                    for _ in range(params['iter']):
                        for batch in create_batch(embeddings, params['batchsize']):
                            ahat, Res = FISTA(torch.from_numpy(batch).to(device), D, args.lda, 100)

                            HessianDiag = HessianDiag.mul((ACT_HISTORY_LEN-1.0)/ACT_HISTORY_LEN) + torch.pow(ahat,2).mean(1)/ACT_HISTORY_LEN

                            D = quadraticBasisUpdate(D, Res, ahat, 0.001, HessianDiag, 0.005) #Dictionary Update
                            progress.update(batch.shape[1])
                    D = D.detach().cpu().numpy()

                np.save(dict_file, D)
                logging.info("Dictionary learning completed...")
            else:
                logging.info('Dictionary file already exists')
        elif D is None and dict_file.startswith('random:'):
            seed = int(dict_file.split(':')[1])
            dict_file = f'{in_file}_norm{args.normalize}_K{args.K}_lda{args.lda}_rnd{seed}'
            np.random.seed(seed)
            D = col_normalize(embeddings @ np.random.randn(embeddings.shape[1], args.K)).astype(embeddings.dtype)
            dd = D.T @ D
            ddd = [dd[i,j] for i in range(dd.shape[0]) for j in range(dd.shape[1]) if i>j]
            logging.info((np.mean(ddd), np.std(ddd), np.min(ddd), np.max(ddd)))
            np.save(dict_file, D)
            logging.info(f'Random dictionary generated using {dict_file}.')

        alphas_file = f'{dict_file}_{os.path.basename(in_file)}_norm{args.normalize}_K{args.K}_lda{args.lda}'
        logging.info((dict_file, alphas_file))

        D = np.load(f'{dict_file}.npy')
        logging.info((D.dtype, embeddings.dtype, D.shape, embeddings.shape))

        if args.spams:
            if not np.isfortran(D):
                D=np.asfortranarray(D)
            alphas = spams.lasso(embeddings, D=D, **lasso_params)
        else:
            alphas_file += '_torch'
            alphas = []
            device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
            D = torch.from_numpy(D).to(device)
            progress = tqdm(range(embeddings.shape[1]))
            for batch in create_batch(embeddings, 4096):
                ahat, _ = FISTA(torch.from_numpy(batch).to(device), D, args.lda, 100)
                alphas.append(scipy.sparse.csc_matrix(ahat.detach().cpu().numpy()))
                progress.update(batch.shape[1])
            alphas = scipy.sparse.hstack(alphas)
        scipy.sparse.save_npz(alphas_file, alphas.T)
        logging.info((alphas_file, alphas.nnz, alphas.shape, alphas.nnz/alphas.shape[1]))

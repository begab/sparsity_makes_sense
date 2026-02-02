import os, sys

import numpy as np

from utils.readers import *

import argparse
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class Preprocessor(object):

    def __init__(self, dataset_reader, transformer, tokenizer_id, gpu, pooling, mlm, mask=False):
        klass = globals()[dataset_reader]
        self.reader = klass(transformer, tokenizer_id, gpu, pooling, mlm, mask)
        self.transformer_model = transformer.split('/')[-1]


    def extract_embeddings(self, in_file_path, out_path, limit=-1, average=False, reduced=False, layers=None):
        """
        Parameters
        ----------
        reduced : bool
          it indicates if we want to store the tagged words only (useful for efficiency purposes)
        layers : list or set of ints
          which layers to save to disk (useful for efficiency purposes)
        """
        logging.info(f'Averaging: {average}. Using {self.transformer_model} and {type(self.reader).__name__} for file {in_file_path}.')

        if layers is None:
            layers = set()
        elif type(layers) is int:
            layers = set([layers])
        elif type(layers) is list:
            layers = set(layers)

        out_file_prefix = None
        if out_path:
            out_file_prefix = f'{out_path}/{os.path.basename(in_file_path)}_{self.transformer_model}_avg_{average}_layer_'
        self.vectors = {}
        self.need_to_open = {}
        id_to_check = None
        layers_list = []
        for i, (embeddings, sequence, is_tagged) in enumerate(self.reader.read_sequences_with_embeddings(in_file_path, limit, average)):

            for layer_id, embs in enumerate(embeddings):
                if len(layers) > 0 and layer_id not in layers: continue
                if layer_id not in self.vectors:
                    #logging.info(layer_id)
                    id_to_check = layer_id
                    layers_list.append(layer_id)
                    self.vectors[layer_id] = []
                    self.need_to_open[layer_id] = True
                assert (average and len(is_tagged)==1) or (not average and len(sequence)==len(is_tagged))
                self.vectors[layer_id].extend(embs[is_tagged] if reduced else embs)

            if out_file_prefix and len(self.vectors[id_to_check]) > 20000:
                self.dump_embeddings(out_file_prefix)
                logging.info((i, len(self.vectors[id_to_check]), sequence))
            if i%5000==0: logging.info(f"{i} sentences processed")
        if out_file_prefix:
            out_files = self.dump_embeddings(out_file_prefix)

        if len(layers) > 1:
            averaged = np.load(f'{out_file_prefix}{layers_list[0]}.npy')
            averaged_file = f'{out_file_prefix}{layers_list[0]}'
            for l in layers_list[1:]:
                logging.info(f'{out_file_prefix}{l}.npy')
                averaged_file += f'-{l}'
                averaged += np.load(f'{out_file_prefix}{l}.npy')
            out_files.append(averaged_file)
            np.save(averaged_file, averaged / len(layers))
        return out_files if out_file_prefix else np.vstack([self.vectors[layers[0]]])


    def dump_embeddings(self, out_file_prefix):
        out_file_names = []
        for layer_id, vecs in self.vectors.items():
            if len(vecs)==0: continue
            out_file = f'{out_file_prefix}{layer_id}'
            out_file_names.append(out_file)
            if self.need_to_open[layer_id]:
                np.save(out_file, vecs)
            else:
                to_extend = np.load(f'{out_file}.npy')
                np.save(out_file, np.vstack([to_extend, vecs]))
            self.vectors[layer_id] = []
            self.need_to_open[layer_id] = False
        return out_file_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determines contextualized representation for sequences in a flexible manner.')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--tokenizer', required=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reader')
    parser.add_argument('--in_files', nargs='+')
    parser.add_argument('--out_dir', default='./representations/')
    parser.add_argument('--pooling', default='mean', choices='mean-first-last-norm'.split('-'))

    parser.add_argument('--mlm', dest='mlm', action='store_true')
    parser.add_argument('--no-mlm', dest='mlm', action='store_false')
    parser.set_defaults(mlm=False) # whether to use the logit (probabilities) or the hidden states
    
    parser.add_argument('--zca', dest='zca', action='store_true')
    parser.add_argument('--no-zca', dest='zca', action='store_false')
    parser.set_defaults(zca=False) # whether decorrelation to be performed
    
    parser.add_argument('--avg-seq', dest='average', action='store_true')
    parser.add_argument('--not-avg-seq', dest='average', action='store_false')
    parser.set_defaults(average=False) # if True avg. of sequence representations get calculated instead of the per token ones

    parser.add_argument('--reduced', dest='reduced', action='store_true', help='Writes only those vectors for which we have labels for.')
    parser.add_argument('--not-reduced', dest='reduced', action='store_false')
    parser.set_defaults(reduced=False)

    parser.add_argument('--mask', dest='mask', action='store_true', help='Whether apply masking of the labeled words.')
    parser.add_argument('--not-mask', dest='mask', action='store_false')
    parser.set_defaults(mask=False)

    parser.add_argument('--layers', nargs='+', default=None, type=int, help='Which layers of the model to save for later computation.')
    args = parser.parse_args()

    if args.reduced == False and args.mask == True:
        logging.error('Illegal combination of args: --not-reduced and --mask cannot be used together.')
        sys.exit(2)

    if args.tokenizer is None:
        args.tokenizer = args.transformer

    dirname = os.path.dirname(f'{args.out_dir}/{args.transformer.replace("/", "_")}_{"reduced" if args.reduced else "full"}/{args.pooling}/')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logging.info(f'{dirname} created')

    logging.info(args)

    p = Preprocessor(args.reader, args.transformer, args.tokenizer, args.gpu_id, args.pooling, args.mlm, args.mask)
    for i,f in enumerate(args.in_files):
        file_names = p.extract_embeddings(f, dirname, average=args.average, reduced=args.reduced, layers=args.layers)

        if args.zca:
            for j,fn in enumerate(file_names):
                X = np.load(f'{fn}.npy')
                if i == 0: # it is important that the first element in args.in_files is the (SemCor) training dataset
                    mu = np.mean(X, axis=0)
                    X -= mu
                    U, sigmas, _ = np.linalg.svd(np.cov(X.T))
                    zca_trafo = U @ np.diag(1/np.sqrt(sigmas + 1e-7)) @ U.T
                    X_whitened = X @ zca_trafo
                    np.save(f'{fn}_zca', np.vstack((mu, zca_trafo)))


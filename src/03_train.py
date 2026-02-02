import os, sys
import pickle
from utils.utils import transform_atoms
from utils.readers import *

import numpy as np
import scipy.sparse

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


def main():

    parser = argparse.ArgumentParser(description='Performs WSD model training.')

    parser.add_argument('--readers', nargs='+')
    parser.add_argument('--in_files', nargs='+')
    parser.add_argument('--representations', nargs='+')
    parser.add_argument('--dictionary_file')
    parser.add_argument('--out_dir')
    
    parser.add_argument('--norm', dest='norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=False)

    parser.add_argument('--reduced', dest='reduced', action='store_true', help='Use it if the input matrix contains embeddings for the labeled words only')
    parser.add_argument('--not-reduced', dest='reduced', action='store_false')
    parser.set_defaults(reduced=False)

    parser.add_argument('--pairs', dest='pairs', action='store_true', help='Whether to use sparse atom pairs')
    parser.add_argument('--not-pairs', dest='pairs', action='store_false')
    parser.set_defaults(pairs=False)

    parser.add_argument('--lexname', dest='senseid', action='store_false')
    parser.add_argument('--senseid', dest='senseid', action='store_true')
    parser.set_defaults(senseid=True)

    parser.add_argument('--wordnet', dest='wn', action='store_true')
    parser.add_argument('--babelnet', dest='wn', action='store_false')
    parser.set_defaults(wn=True)
    
    args = parser.parse_args()

    if not args.senseid and not args.wn:
        logging.warning("Invalid input combination (i.e. usage of lexnames and babelnet is not allowed)")
        sys.exit(2)

    logging.info(args)
    #out_dir_name = os.path.dirname(args.out_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    D = np.load(args.dictionary_file) if args.dictionary_file else None
    for r, inp, rep in zip(args.readers, args.in_files, args.representations):
        logging.info(r)
        if (r!="WordNetReader" and not os.path.exists(inp)) or not os.path.exists(rep):
            logging.warning(f'Either of the files {inp} or {rep} does not exist')
            continue

        labels_to_freq = []
        labels_to_vecs = {}
        labels_to_ids, ids_to_labels = {}, {}
        klass = globals()[r]
        reader = klass()

        if rep.endswith('.npz'):
            M = scipy.sparse.load_npz(rep)
            logging.info(M.shape)
            if D is not None:
                M = M @ D.T
            elif D is None and args.pairs:
                M = transform_atoms(M, weight=True, use_singletons=True, use_pairs=True)

        elif rep.endswith('.npy'):
            M = np.load(rep)

        idx = 0
        for token in reader.get_tokens(inp, args.wn):
            labels = token[0 if args.senseid else 1]
            vec = None
            if not args.reduced or (args.reduced and len(labels) > 0):
                vec = M[idx] / (M[idx].sum() if args.norm and M[idx].sum() > 0 else 1.0)
                idx += 1
            else:
                continue

            for label in labels:
                if label not in labels_to_ids:
                    label_id = len(labels_to_ids)
                    ids_to_labels[label_id] = label
                    labels_to_ids[label] = label_id
                    labels_to_freq.append(1)
                    labels_to_vecs[label_id] = vec
                else:
                    labels_to_freq[labels_to_ids[label]] += 1
                    labels_to_vecs[labels_to_ids[label]] += vec

            if idx%150000==0: logging.info(f'{idx} tokens processed for {inp}')
            
        #logging.info(labels_to_vecs)
    
        if type(labels_to_vecs[0])==np.ndarray:
            mtx = np.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
        else:
            mtx = scipy.sparse.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    
        logging.info((type(mtx), mtx.shape, M.shape, idx))

        #model_file_name = '__'.join([os.path.basename(fn) for fn in args.representations])
        #with open('models/{model_file_name}.pickle', 'wb') as f:
        with open(f'{args.out_dir}/{os.path.basename(rep)}{"_norm" if args.norm else ""}_D{D is not None}{"_pair" if args.pairs else ""}.pickle', 'wb') as fo:
            pickle.dump((labels_to_ids, labels_to_freq, mtx), fo)


if __name__ == '__main__':
    main()

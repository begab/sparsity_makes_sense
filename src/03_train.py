import os, sys
import pickle
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
    out_dir_name = os.path.dirname(args.out_dir)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    D = np.load(args.dictionary_file) if args.dictionary_file else None
    for r, inp, rep in zip(args.readers, args.in_files, args.representations):
        logging.info(r)
        if (r!="WordNetReader" and not os.path.exists(inp)) or not os.path.exists(rep):
            logging.warning('Either of the files {} or {} does not exist'.format(inp, rep))
            continue

        labels_to_freq = []
        labels_to_vecs = {}
        labels_to_ids, ids_to_labels = {}, {}
        klass = globals()[r]
        reader = klass()

        if rep.endswith('.npz'):
            M = scipy.sparse.load_npz(rep)
            if D is not None:
                M = M @ D.T
        elif rep.endswith('.npy'):
            M = np.load(rep)

        idx = 0
        for token in reader.get_tokens(inp, args.wn):
            labels = token[0 if args.senseid else 1]
            vec = None
            if not args.reduced:
                vec = M[idx]
                idx += 1
            elif args.reduced and len(labels) > 0:
                vec = M[idx]
                idx += 1
            else:
                continue

            if args.norm and vec.sum() > 0: vec /= vec.sum()
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

            if idx%150000==0: logging.info('{} tokens processed for {}'.format(idx, inp))
            
        #logging.info(labels_to_vecs)
    
        if type(labels_to_vecs[0])==np.ndarray:
            mtx = np.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
        else:
            mtx = scipy.sparse.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    
        logging.info((type(mtx), mtx.shape, M.shape, idx))

        #model_file_name = '__'.join([os.path.basename(fn) for fn in args.representations])
        #with open('models/{}.pickle'.format(model_file_name), 'wb') as f:
        with open('{}/{}{}_D{}.pickle'.format(out_dir_name, os.path.basename(rep), '_norm' if args.norm else '', D is not None), 'wb') as fo:
            pickle.dump((labels_to_ids, labels_to_freq, mtx), fo)


if __name__ == '__main__':
    main()

import os
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

    parser = argparse.ArgumentParser(description='Performs WSD experiments.')

    parser.add_argument('--readers', nargs='+')
    parser.add_argument('--in_files', nargs='+')
    parser.add_argument('--representations', nargs='+')
    parser.add_argument('--out_file')
    
    parser.add_argument('--norm', dest='norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=False)


    parser.add_argument('--lexname', dest='senseid', action='store_false')
    parser.add_argument('--senseid', dest='senseid', action='store_true')
    parser.set_defaults(senseid=True)

    args = parser.parse_args()

    dir_name = os.path.dirname(args.out_file)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    labels_to_freq, labels_to_vecs = {}, {}
    labels_to_ids, ids_to_labels = {}, {}
    for r, inp, rep in zip(args.readers, args.in_files, args.representations):
        klass = globals()[r]
        reader = klass()

        if rep.endswith('.npz'):
            M = scipy.sparse.load_npz(rep)
        elif rep.endswith('.npy'):
            M = np.load(rep)

        for i, (vec, token) in enumerate(zip(M, reader.get_tokens(inp))):
            labels = token[0 if args.senseid else 1]
            if args.norm and vec.sum() > 0: vec /= vec.sum()
            for label in labels:
                if label not in labels_to_ids:
                    label_id = len(labels_to_ids)
                    ids_to_labels[label_id] = label
                    labels_to_ids[label] = label_id
                    labels_to_freq[label_id] = 1
                    labels_to_vecs[label_id] = vec
                else:
                    labels_to_vecs[labels_to_ids[label]] += vec
                    labels_to_freq[labels_to_ids[label]] += 1

            if i%150000==0: logging.info('{} tokens processed for {}'.format(i, inp))
            if i<10: logging.info((token, len(labels)))
            
    #logging.info(labels_to_vecs)
    
    if type(labels_to_vecs[0])==np.ndarray:
        mtx = np.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    else:
        mtx = scipy.sparse.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    
    logging.info((type(mtx), mtx.shape))

    #model_file_name = '__'.join([os.path.basename(fn) for fn in args.representations])
    #with open('models/{}.pickle'.format(model_file_name), 'wb') as f:
    with open('{}{}.pickle'.format(args.out_file, '_norm' if args.norm else ''), 'wb') as f:
        pickle.dump((labels_to_ids, labels_to_freq, mtx), f)


if __name__ == '__main__':
    main()

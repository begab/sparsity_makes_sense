import os

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

    def __init__(self, dataset_reader, transformer, gpu):
        klass = globals()[dataset_reader]
        self.reader = klass(transformer, gpu)
        self.transformer_model = transformer.split('/')[-1]


    def extract_embeddings(self, in_file_path, out_path, limit=-1, average=False):
        logging.info('Averaging: {}. Using {} and {} for file {}.'.format(average,
            self.transformer_model, type(self.reader).__name__, in_file_path))

        out_file_prefix = '{}/{}_{}_avg_{}_layer_'.format(out_path, os.path.basename(in_file_path), self.transformer_model, average)
        self.vectors = {}
        self.need_to_open = {}
        for i, (embeddings,sequence) in enumerate(self.reader.read_sequences_with_embeddings(in_file_path, limit, average)):

            if embeddings is None:
                for vecs in self.vectors.values():
                    vecs.append(np.zeros(vecs[-1].shape))
            else:
                for layer_id, embs in enumerate(embeddings):
                    if layer_id not in self.vectors:
                        self.vectors[layer_id] = []
                        self.need_to_open[layer_id] = True
                    self.vectors[layer_id].extend(embs)
            if len(self.vectors[0]) > 100000:
                self.dump_embeddings(out_file_prefix)
                logging.info((i, len(self.vectors[0]), sequence))
            if i%5000==0: logging.info("{} sentences processed".format(i))
        self.dump_embeddings(out_file_prefix)

        layers_to_avg = 4
        averaged = np.load('{}{}.npy'.format(out_file_prefix, len(self.vectors) - layers_to_avg))
        averaged_file = '{}{}'.format(out_file_prefix, len(self.vectors) - layers_to_avg)
        layers_averaged = 1
        for l in range(len(self.vectors) - layers_to_avg + 1, len(self.vectors)):
            layers_averaged += 1
            logging.info('{}{}.npy'.format(out_file_prefix, l))
            averaged_file += '-{}'.format(l)
            averaged += np.load('{}{}.npy'.format(out_file_prefix, l))
        logging.info(layers_averaged)
        np.save('{}.npy'.format(averaged_file), averaged / layers_averaged)


    def dump_embeddings(self, out_file_prefix):
        for layer_id in range(len(self.vectors)):
            if len(self.vectors[layer_id])==0: continue
            out_file = '{}{}'.format(out_file_prefix, layer_id)
            if self.need_to_open[layer_id]:
                np.save(out_file, self.vectors[layer_id])
            else:
                to_extend = np.load('{}.npy'.format(out_file))
                np.save(out_file, np.vstack([to_extend, self.vectors[layer_id]]))
            self.vectors[layer_id] = []
            self.need_to_open[layer_id] = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determines contextualized representation for sequences in a flexible manner.')
    parser.add_argument('--transformer', required=True) #, choices=['bert-large-cased', 'bert-large-uncased', 'bert-base-cased', 'bert-base-multilingual-cased', 'xlnet-large-cased'] + ['albert-{}-v2'.format(s) for s in 'base-large-xlarge-xxlarge'.split('-')])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reader')
    parser.add_argument('--in_files', nargs='+')
    parser.add_argument('--out_dir', default='./representations/')

    parser.add_argument('--avg-seq', dest='average', action='store_true')
    parser.add_argument('--not-avg-seq', dest='average', action='store_false')
    parser.set_defaults(average=False) # if True avg. of sequence representations get calculated instead of the per token ones

    args = parser.parse_args()

    dirname = os.path.dirname('{}/{}/'.format(args.out_dir, args.transformer.split('/')[-1]))
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logging.info('{} created'.format(dirname))

    logging.info(args)

    p = Preprocessor(args.reader, args.transformer, args.gpu_id)
    for f in args.in_files:
        p.extract_embeddings(f, dirname, average=args.average)

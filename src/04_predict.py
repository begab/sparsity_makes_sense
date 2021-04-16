import os
import sys
import pickle
import subprocess
from utils.readers import *
from utils.evaluate_answers import parse_file, evaluate
from utils.utils import row_normalize, get_synsets

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from nltk.corpus import wordnet as wn
try:
    wn.get_version()
except:
    import nltk
    nltk.download('wordnet')

import argparse
import itertools
import logging
import logging.config
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s\t%(asctime)s\t%(levelname)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class Evaluator(object):

  def __init__(self, labels_to_indices, freqs, pmi_params):
      self.label_to_id = labels_to_indices
      self.id_to_label = {v:k for k,v in labels_to_indices.items()}
      self.id_to_freq = freqs
      self.use_pmi = pmi_params[0] # do we use PMI (True) or simple averaging (False)
      self.nonneg_pmi = pmi_params[1] # do we require PMIs to be nonnegative (True)
      self.norm_pmi = pmi_params[2] # do we normalize PMI values


  def get_sense_inventory(self, sense_inventory_file, pwn):
      self.sense_inventory = {}
      for line in open(sense_inventory_file):
          if pwn:
              lemma, pos, *rest = line.strip().split()
              self.sense_inventory['{}.{}'.format(lemma, pos).lower()] = rest  # lemma to potential sensekey list
          else:
              lemma, *senses = line.strip().split()
              self.sense_inventory[lemma.lower()] = senses

  def load_model(self, M):
      if type(M) == np.ndarray:
          if self.use_pmi:
              self.M = self.dense_pmi(M)
          else:
              for i,row in enumerate(M):
                  M[i] /= self.id_to_freq[i]
              self.M = row_normalize(np.vstack([M, np.zeros(M.shape[1])])) # reserve an extra row of all zeros for missing categories
      else:
          if self.use_pmi:
              total, row_sum, col_sum = M.sum(), M.sum(axis=1), M.sum(axis=0)
              data, indices, ind_ptr = [], [], [0]
              for i, r in enumerate(M):
                  if np.any(r.data==0):
                      zero_idx = np.where(r.data==0)[0]
                      #logging.warning(("contains 0: ",i,self.id_to_label[i], [r.indices[z] for z in zero_idx]))
                  idxs, pmi_values = self.sparse_pmi(r.indices, r.data, row_sum[i,0], col_sum[0, r.indices], total)
                  indices.extend(idxs)
                  data.extend(pmi_values)
                  ind_ptr.append(len(data))
              ind_ptr.append(len(data)) # reserve an extra row of all zeros for missing categories
              self.M = csr_matrix((data, indices, ind_ptr), shape=(M.shape[0]+1, M.shape[1]))
          else:
              denominators = [self.id_to_freq[i] for i,(f,to) in enumerate(zip(M.indptr, M.indptr[1:])) for _ in range(to-f)]
              M.data /= denominators
              M.indptr = np.hstack((M.indptr, M.indptr[-1])) # reserve an extra row of all zeros for missing categories
              M._shape = (M.shape[0]+1, M.shape[1])
              self.M = M

  def sparse_pmi(self, indices, vals, row_marginal, col_marginal, total):
      row_marginal += 1e-11
      col_marginal += 1e-11
      pmis = np.ma.log((total * vals) / (row_marginal * col_marginal)).filled(0)
      if self.norm_pmi:
          pmis /= -np.ma.log(vals/total).filled(1)
      indices_to_return, pmis_to_return = [], []
      for idx in range(len(indices)):
          if not self.nonneg_pmi or pmis[0,idx] > 0:
              indices_to_return.append(indices[idx])
              pmis_to_return.append(pmis[0,idx])
      return indices_to_return, pmis_to_return

  def dense_pmi(self, incidence_mtx):
      total_observations = incidence_mtx.sum()
      row_marginals = np.sum(incidence_mtx, axis=1) / total_observations
      col_marginals = np.sum(incidence_mtx, axis=0) / total_observations
      joint_probs = incidence_mtx / total_observations
      outerP = np.outer(row_marginals, col_marginals)

      # masked operations allow for avoiding numeric issues, e.g dividing by 0
      pmi = np.ma.log(np.ma.divide(outerP, joint_probs).filled(1)).filled(0)
      if self.norm_pmi:
          pmi = np.ma.divide(pmi, np.ma.log(joint_probs).filled(0)).filled(0)
      if self.nonneg_pmi:
          pmi[pmi<0] = 0
      return pmi

  def lemma_based_sense_selection(self, lemma, pwn):
      potential_senses, potential_synsets, potential_lexnames = [], [], []
      for s in self.sense_inventory[lemma.lower()]:
          synset_name = synset_lexname = s
          if pwn:
              synset = wn.lemma_from_key(s).synset()
              synset_name = synset.name()
              synset_lexname = synset.lexname()
          potential_senses.append(s)
          potential_synsets.append(synset_name)
          potential_lexnames.append(synset_lexname)
      return potential_senses, potential_synsets, potential_lexnames


  def token_based_sense_selection(self, raw_token):
      potential_senses, potential_synsets, potential_lexnames = [], [], []
      for synset in get_synsets(raw_token)[0]:
          for lemma in synset.lemmas():
              potential_senses.append(lemma.key())
              potential_synsets.append(synset.name())
              potential_lexnames.append(synset.lexname())
      return potential_senses, potential_synsets, potential_lexnames

  
  def print_predictions(self, out_file, ids, preds, filter_for=None):
      with open(out_file, 'w') as f:
          for lemma_id, pred in zip(ids, preds):
              if filter_for and filter_for != 'ALL' and lemma_id.startswith(filter_for):
                  f.write('{} {}\n'.format(lemma_id.replace('{}.'.format(filter_for), ''), pred))
              elif filter_for is None or filter_for == 'ALL':
                  f.write('{} {}\n'.format(lemma_id, pred))


def main():
    parser = argparse.ArgumentParser(description='Performs evaluation.')

    parser.add_argument('--reader', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_inputs', nargs='+', type=str, required=True)
    parser.add_argument('--eval_repr', type=str, required=True)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--dictionary_file', type=str)
    parser.add_argument('--inventory_file', type=str, default='/Data_Validation/candidatesWN30.txt')

    parser.add_argument('--reduced', dest='reduced', action='store_true', help='Use it if the input matrix contains embeddings for the labeled words only')
    parser.add_argument('--not-reduced', dest='reduced', action='store_false')
    parser.set_defaults(reduced=False)
 

    parser.add_argument('--batch_experiment', dest='single_experiment', action='store_false')
    parser.set_defaults(single_experiment=True)

    parser.add_argument('--use_pmi', dest='use_pmi', action='store_true')
    parser.set_defaults(use_pmi=False)

    parser.add_argument('--nonneg_pmi', dest='nonneg_pmi', action='store_true')
    parser.set_defaults(nonneg_pmi=False)

    parser.add_argument('--normalize_pmi', dest='normalize_pmi', action='store_true')
    parser.set_defaults(normalize_pmi=False)

    parser.add_argument('--discard_lemma_info', dest='lemma_info', action='store_false')
    parser.set_defaults(lemma_info=True)
    
    parser.add_argument('--lexname', dest='senseid', action='store_false')
    parser.add_argument('--senseid', dest='senseid', action='store_true')
    parser.set_defaults(senseid=True)

    parser.add_argument('--wordnet', dest='use_pwn', action='store_true')
    parser.add_argument('--babelnet', dest='use_pwn', action='store_false')
    parser.set_defaults(use_pwn=True)


    args = parser.parse_args()

    klass = globals()[args.reader]
    reader = klass()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    labels_to_freq, labels_to_vecs, labels_to_idx = [], {}, {}
    for mi in args.model_inputs:
        lab_to_idx, freqs, vecs = pickle.load(open(mi, 'rb'))
        for l,idx in lab_to_idx.items():
            freq = freqs[idx]
            if l not in labels_to_idx:
                label_id = len(labels_to_idx)
                labels_to_idx[l] = label_id
                labels_to_freq.append(freq)
                labels_to_vecs[label_id] = vecs[idx]
            else:
                labels_to_freq[labels_to_idx[l]] += freq
                labels_to_vecs[labels_to_idx[l]] += vecs[idx]

    if type(labels_to_vecs[0])==np.ndarray:
        mtx = np.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    else:
        mtx = scipy.sparse.vstack([labels_to_vecs[row] for row in sorted(labels_to_vecs)])
    model_name = '&'.join([os.path.basename(fn) for fn in args.model_inputs])
    #logging.info((len(labels_to_vecs), len(labels_to_freq), labels_to_freq[0:10], mtx.shape, model_name))

    D = np.load(args.dictionary_file) if args.dictionary_file else None

    for b1,b2,b3 in itertools.product([True, False], repeat=3):
        if args.single_experiment and (b1 != args.use_pmi or b2 != args.nonneg_pmi or b3 != args.normalize_pmi):
            continue

        params = [b1, b2, b3]

        if not b1 and (b2 or b3):
            if args.single_experiment:
                logging.warning('When PMI is not used the PMI specific parameters (--nonneg_pmi and --normalize) should not be set.')
            continue

        ev = Evaluator(labels_to_idx, labels_to_freq, params)
        ev.load_model(mtx)

        if args.reader == 'SemcorReader':
            if args.use_pwn:
                if not os.path.exists("{}/Evaluation_Datasets/Scorer.class".format(args.eval_dir)):
                    subprocess.run(["javac", "{}/Evaluation_Datasets/Scorer.java".format(args.eval_dir)])
            ev_dir = args.eval_dir
            ev.get_sense_inventory('{}/{}'.format(ev_dir, args.inventory_file), args.use_pwn)

        if args.eval_repr.endswith('npy'):
            R = np.load(args.eval_repr)
        elif args.eval_repr.endswith('npz'):
            R = scipy.sparse.load_npz(args.eval_repr)
            if D is not None:
                R = R @ D.T

        predictions = {}
        ids, preds, expected = [], [], []
        for i, token in enumerate(reader.get_tokens(args.input_file, args.use_pwn)):

            if len(token[0])==0: continue  # skip tokens that are not labeled
            vec = R[len(preds) if args.reduced else i]

            possible_indices = range(len(ev.label_to_id))
            if args.reader == 'SemcorReader' and args.senseid:
                token_id = token[2]
                lemma, raw_word = token[3:]
                if args.lemma_info:
                    possible_labels, possible_synsets, possible_lexnames = ev.lemma_based_sense_selection(lemma, args.use_pwn)
                elif args.use_pwn:
                    possible_labels, possible_synsets, possible_lexnames = ev.token_based_sense_selection(raw_word)
                else: # conduct a totally unconditioned prediction
                    possible_labels = ev.id_to_label
                    possible_synsets = possible_labels
                # logging.info((lemma, raw_word, possible_labels, possible_synsets, R.shape, type(R)))

                synset_indices = [ev.label_to_id[s] if s in ev.label_to_id else -1 for s in possible_synsets]

                if len(synset_indices)==0:
                    logging.warning('Unable to predict for {}'.format(raw_word))
                    continue
                else:
                    possible_indices = synset_indices

                expected.append(token[0])
                if len(possible_labels)==0:
                    logging.warning('No etalon sensekey for {}'.format(raw_word))
            else:
                token_id = i
                possible_labels = ev.id_to_label
                expected.append(token[0 if args.senseid else 1][0])

            ids.append(token_id)
            scores = ev.M[possible_indices] @ vec.T
            preds.append(possible_labels[np.argmax(scores)])
            predictions[token_id] = set([preds[-1]])

        with open('./outputs/{}_{}.out'.format('_'.join(map(str, params)), np.abs(hash(model_name))), 'w') as fo:
            fo.write('\n'.join(preds))

        if args.reader == 'SemcorReader':
            if args.use_pwn:
                for d in os.listdir('{}/Evaluation_Datasets'.format(ev_dir)):
                    if not os.path.isdir('{}/Evaluation_Datasets/{}'.format(ev_dir, d)): continue

                    tmp_pred_file = '{}_tmp.key'.format(np.abs(hash(model_name)))
                    ev.print_predictions(tmp_pred_file, ids, preds, d)
                    result=subprocess.check_output(["java", "-cp", "{}/Evaluation_Datasets".format(ev_dir), "Scorer", "{}/Evaluation_Datasets/{}/{}.gold.key.txt".format(ev_dir, d,d), tmp_pred_file])
                    p_r_f = result.decode('utf-8').split()
                    prf = '\t'.join([p_r_f[i].replace('%', '') for i in [1,3,5]])
                    print('{}\t{}\t{}\t{}\t{}'.format(len(preds), '\t'.join(map(str, params)), prf, d, model_name))
                    os.remove(tmp_pred_file)
                print("================")
            else:
                gold = parse_file(args.input_file.replace('data.xml', 'gold.key.txt'))
                score = evaluate(predictions, gold, False)
                for k,v in score.items():
                    print('{}\t{}\t{:.4f}\t{}\t{}'.format(len(preds), '\t'.join(map(str, params)), v, k, os.path.basename(args.eval_repr)))
        else:
            if len(expected) != len(preds):
                logging.warning('There is a mismatch in the number of gold annotations and predictions.')
            accuracy = sum([1 if x[0]==x[1] else 0 for x in zip(preds, expected)]) / len(preds)
            logging.info("{}\t{}\tAccuracy: {:.5f}".format(model_name, '\t'.join(map(str, params)), accuracy))

if __name__ == '__main__':
    main()

import os
import sys
import pickle
import subprocess
from utils.readers import *
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
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class Evaluator(object):

  def __init__(self, model_file, pmi_params):
      self.model_file = model_file
      self.use_pmi = pmi_params[0] # do we use PMI (True) or simple averaging (False)
      self.nonneg_pmi = pmi_params[1] # do we require PMIs to be nonnegative (True)
      self.norm_pmi = pmi_params[2] # do we normalize PMI values


  def get_sense_inventory(self, sense_inventory_file):
      self.sense_inventory = {}
      for line in open(sense_inventory_file):
          lemma, pos, *rest = line.strip().split()
          self.sense_inventory['{}.{}'.format(lemma, pos)] = rest  # lemma to potential sensekey list


  def load_model(self):
      with open(self.model_file, 'rb') as mf:
          model = pickle.load(mf)
      self.label_to_id = model[0]
      self.id_to_label = {v:k for k,v in model[0].items()}
      self.id_to_freq = model[1]
      M = model[2]

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

  def lemma_based_sense_selection(self, lemma):
      potential_senses, potential_synsets, potential_lexnames = [], [], []
      for s in self.sense_inventory[lemma]:
          synset = wn.lemma_from_key(s).synset()
          potential_senses.append(s)
          potential_synsets.append(synset.name())
          potential_lexnames.append(synset.lexname())
      return potential_senses, potential_synsets, potential_lexnames


  def token_based_sense_selection(self, raw_token):
      potential_synsets, potential_senses, potential_lexnames = [], [], []
      for synset in get_synsets(raw_token)[0]:
          synset_id = self.synset_to_id[synset.name()]
          for lemma in synset.lemmas():
              potential_synsets.append(synset_id)
              potential_senses.append(lemma.key())
              potential_lexnames.append(synset.lexname())
      return potential_synsets, potential_senses, potential_lexnames

  
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
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--eval_repr', type=str, required=True)
    parser.add_argument('--eval_dir', type=str)
    parser.add_argument('--inventory_file', type=str, default='/Data_Validation/candidatesWN30.txt')

    parser.add_argument('--batch_experiment', dest='single_experiment', action='store_false')
    parser.set_defaults(single_experiment=True)

    parser.add_argument('--use_pmi', dest='use_pmi', action='store_true')
    parser.set_defaults(use_pmi=False)

    parser.add_argument('--nonneg_pmi', dest='nonneg_pmi', action='store_true')
    parser.set_defaults(nonneg_pmi=False)

    parser.add_argument('--normalize_pmi', dest='normalize_pmi', action='store_true')
    parser.set_defaults(normalize_pmi=False)

    parser.add_argument('--dismiss_lemma_info', dest='lemma_info', action='store_false')
    parser.set_defaults(lemma_info=True)
    
    parser.add_argument('--xling', dest='english_data', action='store_false')
    parser.set_defaults(english_data=True)
    args = parser.parse_args()

    klass = globals()[args.reader]
    reader = klass()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    for b1,b2,b3 in itertools.product([True, False], repeat=3):
        if args.single_experiment and (b1 != args.use_pmi or b2 != args.nonneg_pmi or b3 != args.normalize_pmi):
            continue

        params = [b1, b2, b3]

        if not b1 and (b2 or b3):
            if args.single_experiment:
                logging.warning('When PMI is not used the PMI specific parameters (--nonneg_pmi and --normalize) should not be set.')
            continue

        ev = Evaluator(args.model_file, params)
        ev.load_model()

        if args.reader == 'SemcorReader':
            if not os.path.exists("{}/Evaluation_Datasets/Scorer.class".format(args.eval_dir)):
                subprocess.run(["javac", "{}/Evaluation_Datasets/Scorer.java".format(args.eval_dir)])
            ev_dir = args.eval_dir
            ev.get_sense_inventory('{}/{}'.format(ev_dir, args.inventory_file))

        if args.eval_repr.endswith('npy'):
            R = np.load(args.eval_repr)
        elif args.eval_repr.endswith('npz'):
            R = scipy.sparse.load_npz(args.eval_repr)

        ids, preds, expected = [], [], []
        for i, (token, vec) in enumerate(zip(reader.get_tokens(args.input_file, args.english_data), R)):

            if len(token[0])==0: continue  # skip tokens that are not labeled

            possible_indices = range(len(ev.label_to_id))
            if args.reader == 'SemcorReader':
                token_id = token[2]
                lemma, raw_word = token[3:]
                if args.lemma_info:
                    possible_labels, possible_synsets, _ = ev.lemma_based_sense_selection(lemma)
                elif args.english_data:
                    possible_labels, possible_synsets, _ = ev.token_based_sense_selection(raw_word)
                else:
                    pass
                # logging.info((lemma, raw_word, possible_labels, R.shape, type(R)))

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
                expected.append(token[0][0])

            ids.append(token_id)
            scores = ev.M[possible_indices] @ vec.T
            preds.append(possible_labels[np.argmax(scores)])

        with open('./outputs/{}_{}.out'.format('_'.join(map(str, params)), os.path.basename(args.model_file)), 'w') as fo:
            fo.write('\n'.join(preds))

        if args.reader == 'SemcorReader':
            if args.english_data:
                for d in os.listdir('{}/Evaluation_Datasets'.format(ev_dir)):
                    if not os.path.isdir('{}/Evaluation_Datasets/{}'.format(ev_dir, d)): continue

                    tmp_pred_file = '{}_tmp.key'.format(args.model_file)
                    ev.print_predictions(tmp_pred_file, ids, preds, d)
                    result=subprocess.check_output(["java", "-cp", "{}/Evaluation_Datasets".format(ev_dir), "Scorer", "{}/Evaluation_Datasets/{}/{}.gold.key.txt".format(ev_dir, d,d), tmp_pred_file])
                    p_r_f = result.decode('utf-8').split()
                    prf = '\t'.join([p_r_f[i].replace('%', '') for i in [1,3,5]])
                    print('{}\t{}\t{}\t{}\t{}'.format(len(preds), '\t'.join(map(str, params)), prf, d, args.model_file))
                    os.remove(tmp_pred_file)
                print("================")
            else:
                #print(ids[0:10], preds[0:10], len(ids), len(preds))
                correct, total = 0, 0
                for i,(p,e) in enumerate(zip(preds, expected)):
                    synset_name = wn.lemma_from_key(p).synset().name()
                    if synset_name in e:
                        correct += 1
                    total += len(e)
                prec, rec = correct / len(expected), correct /total
                logging.info('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}'.format(args.model_file, '\t'.join(map(str, params)), prec, rec, 2*prec*rec / (prec+rec), correct))
        else:
            if len(expected) != len(preds):
                logging.warning('There is a mismatch in the number of gold annotations and predictions.')
            accuracy = sum([1 if x[0]==x[1] else 0 for x in zip(preds, expected)]) / len(preds)
            logging.info("{}\t{}\tAccuracy: {:.5f}".format(args.model_file, '\t'.join(map(str, params)), accuracy))

if __name__ == '__main__':
    main()

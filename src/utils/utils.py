import numpy as np
import collections

import xml.etree.ElementTree as ET

from nltk.corpus import wordnet as wn
try:
    wn.get_version()
except:
    import nltk
    nltk.download('wordnet')

import nltk.data
try:
    nltk.data.load('tokenizers/punkt/english.pickle')
except:
    import nltk
    nltk.download('punkt')

def row_normalize(embeddings):
    row_norms = np.sqrt((embeddings**2).sum(axis=1))[:, np.newaxis]
    row_norms[row_norms==0] = 1 # we do not want to divide by 0
    return embeddings / row_norms

def col_normalize(embeddings):
    col_norms = np.sqrt((embeddings**2).sum(axis=0))[np.newaxis,:]
    col_norms[col_norms==0] = 1 # we do not want to divide by 0
    return embeddings / col_norms

def write_embeddings(embeddings, words, out_file_name, new_file=False):
    if type(embeddings) == list:
        embeddings = np.array(embeddings)
    dense = type(embeddings) == np.ndarray
    dim = embeddings.shape[1 if dense else 0]
    with open(out_file_name, 'w' if new_file else 'a') as f:
        for i,w in enumerate(words):
            if dense:
                to_print = embeddings[i]
            else:
                c = embeddings.getcol(i)
                to_print = collections.defaultdict(int, zip(c.indices, c.data))
            f.write('{} {}\n'.format(w, ' '.join(map(str, [round(to_print[j],6) for j in range(dim)]))))

def process_raganato_data(file_path):
    '''
    Maps the within document token position to the lemma&POS info form of the token.
    '''
    root = ET.parse(file_path).getroot()
    position_to_lemmas={}
    position_to_ids = {}
    tokens = []
    c=-1
    for s in root.findall('text/sentence'):
        for token in list(s):
            c+=1
            tokens.append(token.text)
            pos_tag = token.attrib['pos']
            pos_tag_wn = 'r'
            if pos_tag!="ADV": pos_tag_wn = pos_tag[0].lower()
            position_to_lemmas[c] = ('{}.{}'.format(token.attrib['lemma'], pos_tag_wn),
                                     token.text.replace('-', '_'))
            if 'id' in token.attrib:
                position_to_ids[c] = token.attrib['id']
    return position_to_lemmas, position_to_ids, tokens

def process_raganato_gold(key_file):
    id_to_gold, sense_to_id = {}, {}
    with open(key_file) as f:
          for l in f:
            position, *rest=l.split()
            id_to_gold[position] = rest

            for s in rest:
                if s not in sense_to_id:
                    sense_to_id[s] = [len(sense_to_id), 1]
                else:
                    sense_to_id[s][1] += 1
    return id_to_gold, sense_to_id

def get_synsets(raw_word, pos=None):
    """
    Tries out a few simple normalization strategies to bind words to synsets with higher recall.
    """
    for w in [raw_word, raw_word.replace('_', '-'), raw_word.replace('_', ''), raw_word.replace(' ', ''), raw_word.split()[0]]:
        synsets = wn.synsets(w.replace(' ', '_'))
        if pos:
            pos_filtered_synsets = [s for s in synsets if s.pos() == pos]
        else:
            pos_filtered_synsets = synsets
        if len(synsets) > 0:
            return synsets, pos_filtered_synsets
    return list(wn.all_synsets()), []

def load_lexname_mapping():
    lexname_to_id = {}
    for s in wn.all_synsets():
        if s.lexname() not in lexname_to_id:
            lexname_to_id[s.lexname()] = len(lexname_to_id)
    return lexname_to_id

def calculate_column_norms(sparse_matrix):
    """
    Calculates the columnwise norms of a scipy.sparse.csr (or csc) matrix in an efficient manner.
    """
    squared_norms = collections.defaultdict(float)
    for s, value in zip(sparse_matrix.indices, sparse_matrix.data):
        squared_norms[s] += value**2
    return {k:np.sqrt(v) for k,v in squared_norms.items()}


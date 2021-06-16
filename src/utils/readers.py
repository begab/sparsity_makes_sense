import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nltk import word_tokenize
from nltk.corpus import wordnet as wn
try:
    wn.get_version()
except:
    import nltk
    nltk.download('wordnet')

import json
import xml.etree.ElementTree as ET

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


class SeqReader(object):

    def __init__(self, transformer=None, gpu=0, pooling='mean'):
        self.transformer_model = transformer
        self.pooling_strategy = pooling
        if transformer is not None:
            self.tokenizer, self.model = self.load_transformer(transformer, gpu)

    def set_device(self, device_id):
        device_count = torch.cuda.device_count()
        if device_count != 0:
            if device_id < 0 or device_id >= device_count:
                device_id = random.randint(1, device_count) - 1
            self.device = torch.device('cuda:{}'.format(device_id))
        else:
            self.device = torch.device('cpu')

    def load_transformer(self, transformer, gpu_id):
        self.set_device(gpu_id)
        conf = AutoConfig.from_pretrained(transformer, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(transformer)
        model = AutoModel.from_pretrained(transformer, config=conf)
        model.to(self.device)
        return tokenizer, model

    def read_sequences(self, in_file, limit=-1):
        raise Exception("Method unimplemented")

    def read_sequences_with_embeddings(self, in_file, limit=-1, average=False):
        if self.transformer_model is None:
            raise Exception('Reader object was not initialized with a transformer model')

        for i,(sequence, is_tagged) in enumerate(self.read_sequences(in_file, limit)):
            is_tagged = [True] if average else is_tagged
            yield self.process_sequence(sequence, average) + (is_tagged,)

    def process_sequence(self, seq, average):
        tokens, indexed_tokens = self.tokenize_sequence(seq)
        if tokens is None:
            return None, seq
        else:
            return self.get_representation(tokens, indexed_tokens, average), seq

    def tokenize_sequence(self, sequence):
        orig_to_tok_map, transformer_tokens = [], []
        for orig_token in sequence:
            orig_to_tok_map.append(len(transformer_tokens))
            transformer_tokens.extend(self.tokenizer.tokenize(orig_token))

        orig_to_tok_map.append(len(transformer_tokens))

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(transformer_tokens)
        indexed_tokens_with_specials = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)

        if len(indexed_tokens) == 0:
            logging.warning(("Indexed tokens have 0 length: ", sequence))
            return None, None

        specials_added = indexed_tokens_with_specials.index(indexed_tokens[0])
        orig_to_tok_map = [x + specials_added for x in orig_to_tok_map]
        return orig_to_tok_map, indexed_tokens_with_specials


    def get_representation(self, orig_to_tok_map, indexed_tokens_with_specials, average):
        with torch.no_grad():
            vecs = self.model(torch.tensor([indexed_tokens_with_specials]).to(self.device))['hidden_states']

        per_layer_embeddings = []
        for emb in vecs:
            if average:
                averaged = torch.mean(emb[0], dim=0).detach().cpu().numpy().reshape(1,-1)
                per_layer_embeddings.append(averaged)
            else:
                token_embeddings = []
                for k,l in zip(orig_to_tok_map, orig_to_tok_map[1:]):
                    ki, li = k, l
                    if self.pooling_strategy == 'first':
                        li=ki+1
                    elif self.pooling_strategy == 'last':
                        ki=li-1
                    elif self.pooling_strategy == 'norm':
                        norms = torch.linalg.norm(emb[0, k:l], dim=1)
                        ki += torch.argmax(norms).item()
                        li = ki + 1
                    token_embeddings.append(torch.mean(emb[0, ki:li], dim=0).detach().cpu().numpy())
                per_layer_embeddings.append(np.array(token_embeddings))
        return per_layer_embeddings
    

class SemcorReader(SeqReader):

    def read_sequences(self, in_file, limit=-1):
        root = ET.parse(in_file).getroot()
        for i,s in enumerate(root.findall('text/sentence')):
            if i==limit: break

            seq_tokens, is_tagged = [], []
            for orig_token in list(s):
                seq_tokens.append(orig_token.text)
                is_tagged.append(orig_token.tag=='instance')
            yield seq_tokens, is_tagged


    def get_tokens(self, in_file, pwn_labels=True):

        #etalons, _ = self.get_labels(in_file.replace('data.xml', '{}gold.key.txt'.format('' if english else 'wnids.')))
        etalons, _ = self.get_labels(in_file.replace('data.xml', 'gold.key.txt'))
        root = ET.parse(in_file).getroot()
        pos_delim = '.' if pwn_labels else '#'
        for s in root.findall('text/sentence'):
            for token in list(s):
                pos_tag = token.attrib['pos']
                if pwn_labels:
                    normalized_pos = 'r'
                    if len(pos_tag)>0 and pos_tag!="ADV": normalized_pos = pos_tag[0].lower()
                else:
                    normalized_pos = pos_tag
                token_id = None
                synset_labels, lexname_labels = [], []
                if token.tag=='instance' and token.attrib['id'] in etalons:
                    token_id = token.attrib['id']
                    for sensekey in etalons[token_id]:
                        synset = None
                        if pwn_labels:
                            try:
                                wn_lemma = wn.lemma_from_key(sensekey)
                                synset = wn_lemma.synset()
                            except Exception as e:
                                synset = wn.synset_from_sense_key(sensekey)
                                logging.warning("Potential problem with mapping sensekey {} to synset {}".format(sensekey, synset))
                                # see the issue https://github.com/nltk/nltk/issues/2171 and the PR https://github.com/nltk/nltk/pull/2621
                            if synset is not None:
                                synset_labels.append(synset.name())
                                lexname_labels.append(synset.lexname())
                        else:
                            # in the pre XL-WSD era we used to do the following:
                            # synset = wn.synset_from_pos_and_offset(sensekey[-1], int(sensekey[3:-1]))
                            synset_labels.append(sensekey)
                if 'lemma' in token.attrib:
                    lemma = '{}{}{}'.format(token.attrib['lemma'], pos_delim,  normalized_pos)
                else:
                    lemma = '{}{}{}'.format(token.text, pos_delim, normalized_pos)
                yield synset_labels, lexname_labels, token_id, lemma, token.text.replace('-', '_')


    def get_labels(self, key_file):
        id_to_gold, sense_to_id = {}, {}
        with open(key_file) as f:
            for l in f:
                position_id, *senses = l.split()
                id_to_gold[position_id] = senses
                
                for s in senses:
                    if s not in sense_to_id:
                        sense_to_id[s] = [len(sense_to_id), 1]
                    else:
                        sense_to_id[s][1] += 1
        return id_to_gold, sense_to_id


class WngtReader(SeqReader):

    def read_sequences(self, in_file, limit=-1):
        root = ET.parse(in_file).getroot()
        for i,s in enumerate(root.findall('document/paragraph/sentence')):
            if i==limit: break

            seq_tokens, is_tagged = [], []
            for orig_token in list(s):
                seq_tokens.append(orig_token.attrib['surface_form'].replace('_', ' '))
                is_tagged.append('wn30_key' in orig_token.attrib)
            yield seq_tokens, is_tagged

    def get_tokens(self, in_file):
        root = ET.parse(in_file).getroot()
        for i,s in enumerate(root.findall('document/paragraph/sentence')):
            for t in s:
                synset_labels, lexname_labels = [], []
                if 'wn30_key' in t.attrib:
                    sensekey = t.attrib['wn30_key']
                    try:
                        synset = wn.lemma_from_key(sensekey).synset()
                    except Exception as e:
                        sensekey = sensekey.replace('%3', '%5') # a fix for unprocessable satellites
                        synset = wn.lemma_from_key(sensekey).synset() # now, we should be able to find the modified sensekey in WN
                    synset_labels.append(synset.name())
                    lexname_labels.append(synset.lexname())
                yield synset_labels, lexname_labels, t.attrib['surface_form']


class WordNetReader(SeqReader):


    def read_sequences(self, in_file=None, limit=-1):
        data = []
        for i, synset in enumerate(wn.all_synsets()):
            if i==limit: break
            gloss = ' '.join(word_tokenize(synset.definition()))
            all_lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
            d_str = ' , '.join(all_lemmas) + ' - ' + gloss
            data.append((synset, d_str))

        data = sorted(data, key=lambda x: x[0])
        for s in data:
            tokens = s[1].split()
            yield tokens, len(tokens) * [True]

    def get_tokens(self, in_file=None):
        data = []
        for i, synset in enumerate(wn.all_synsets()):
            gloss = ' '.join(word_tokenize(synset.definition()))
            all_lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
            d_str = ' , '.join(all_lemmas) + ' - ' + gloss
            data.append((synset, d_str))

        data = sorted(data, key=lambda x: x[0])
        for s in data:
            yield [s[0].name()], [s[0].lexname()]


class ConllReader(SeqReader):

    def __init__(self, transformer=None, gpu=0, wc=0, lc=1):
        super().__init__(transformer, gpu)
        self.word_column = wc
        self.label_column = lc

    def read_sequences(self, in_file, limit=-1):
        sentence_counter = 0
        sentence = []
        for i,line in enumerate(open(in_file)):
            if len(line.strip())==0 and len(sentence)>0:
                yield sentence, len(sentence) * [True]
                sentence=[]
                sentence_counter += 1
                if sentence_counter==limit: break
            elif len(line.strip())>0:
                sentence.append(line.split()[self.word_column])
        if len(sentence) > 0:
            yield sentence, len(sentence) * [True]

    def get_tokens(self, in_file):
        labeled_tokens = []
        for i,line in enumerate(open(in_file)):
            if len(line.strip())>0:
                parts = line.split()
                yield [parts[self.label_column]], parts[self.word_column]


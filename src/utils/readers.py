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

    def __init__(self, transformer=None, gpu=0):
        self.transformer_model = transformer
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

        for i,sequence in enumerate(self.read_sequences(in_file, limit)):
            yield self.process_sequence(sequence, average)

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
            vecs = self.model(torch.tensor([indexed_tokens_with_specials]).to(self.device))[-1]

        per_layer_embeddings = []
        for emb in vecs:
            if average:
                averaged = torch.mean(emb[0], dim=0).detach().cpu().numpy().reshape(1,-1)
                per_layer_embeddings.append(averaged)
            else:
                token_embeddings = []
                for k,l in zip(orig_to_tok_map, orig_to_tok_map[1:]):
                    token_embeddings.append(torch.mean(emb[0, k:l], dim=0).detach().cpu().numpy())
                per_layer_embeddings.append(np.array(token_embeddings))
        return per_layer_embeddings


class SemcorReader(SeqReader):

    def read_sequences(self, in_file, limit=-1):
        root = ET.parse(in_file).getroot()
        for i,s in enumerate(root.findall('text/sentence')):
            if i==limit: break

            seq_tokens = []
            for orig_token in list(s):
                seq_tokens.append(orig_token.text)
            yield seq_tokens


    def get_tokens(self, in_file, english=True):
        etalons, _ = self.get_labels(in_file.replace('data.xml', '{}gold.key.txt'.format('' if english else 'wnids.')))
        root = ET.parse(in_file).getroot()
        for s in root.findall('text/sentence'):
            for token in list(s):
                pos_tag = token.attrib['pos']
                pos_tag_wn = 'r'
                if pos_tag!="ADV": pos_tag_wn = pos_tag[0].lower()
                token_id = None
                synset_labels, lexname_labels = [], []
                if token.tag=='instance' and token.attrib['id'] in etalons:
                    token_id = token.attrib['id']
                    for sensekey in etalons[token.attrib['id']]:
                        if english:
                            synset = wn.lemma_from_key(sensekey).synset()
                        else:
                            synset = wn.synset_from_pos_and_offset(sensekey[-1], int(sensekey[3:-1]))
                        synset_labels.append(synset.name())
                        lexname_labels.append(synset.lexname())
                if 'lemma' in token.attrib:
                    lemma = '{}.{}'.format(token.attrib['lemma'], pos_tag_wn)
                else:
                    lemma = '{}.{}'.format(token.text, pos_tag_wn)
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

            seq_tokens = []
            for orig_token in list(s):
                token = orig_token.attrib['surface_form'].replace('_', ' ')
                seq_tokens.append(token.replace(' ', '_'))
            yield seq_tokens

    def get_tokens(self, in_file, english=True):
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
            yield s[1].split()

    def get_tokens(self, in_file=None, english=True):
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
                yield sentence
                sentence=[]
                sentence_counter += 1
                if sentence_counter==limit: break
            elif len(line.strip())>0:
                sentence.append(line.split()[self.word_column])
        if len(sentence) > 0:
            yield sentence

    def get_tokens(self, in_file, english=True):
        labeled_tokens = []
        for i,line in enumerate(open(in_file)):
            if len(line.strip())>0:
                parts = line.split()
                yield [parts[self.label_column]], parts[self.word_column]


class WiCReader(SeqReader):
    
    def read_sequences(self, in_file, limit=-1):
        for i,line in enumerate(open(in_file)):
            if i==limit: break
            _, _, _, s1, s2 = line.strip().split('\t')
            yield s1.split()
            yield s2.split()

    def get_tokens(self, in_file, english=True):
        for i,line in enumerate(open(in_file)):
            word, pos, location, s1, s2 = line.strip().split('\t')
            loc1, loc2 = map(int, location.split('-'))
            for l,t in enumerate(s1.split()):
                labels = []
                if l==loc1:
                    labels.append((word, pos))
                yield labels, t
            for l,t in enumerate(s2.split()):
                labels = []
                if l==loc2:
                    labels.append((word, pos))
                yield labels, t

class WiCtsvReader(SeqReader):

    def read_sequences(self, in_file, limit=-1):
        for i,line in enumerate(open(in_file)):
            if i==limit: break
            if 'examples' in in_file:
                word, location, s = line.strip().split('\t')
                yield s.split()
            elif 'definitions' in in_file:
                tokens = line.strip().split()
                yield tokens
            elif 'hypernyms' in in_file:
                tokens = line.strip().split()
                yield [t.replace('_', ' ') for t in tokens]


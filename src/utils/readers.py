import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, DataCollatorWithPadding

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
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

class SeqReader(object):

    def __init__(self, transformer=None, tokenizer_id=None, gpu=0, pooling='mean', mlm=False, mask=False):
        self.transformer_model = transformer
        self.pooling_strategy = pooling
        self.mlm = mlm
        self.mask = mask
        if transformer is not None:
            self.tokenizer, self.model = self.load_transformer(transformer, tokenizer_id, gpu, mlm)
            self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True, return_tensors='pt')

    def set_device(self, device_id):
        device_count = torch.cuda.device_count()
        if device_count != 0 and device_id >= 0:
            if device_id >= device_count:
                device_id = np.random.randint(device_count)
            self.device = torch.device('cuda:{}'.format(device_id))
        else:
            self.device = torch.device('cpu')

    def load_transformer(self, transformer, tokenizer_id, gpu_id, mlm=False):
        self.set_device(gpu_id)
        trust_remote_code=False
        conf = AutoConfig.from_pretrained(transformer, output_hidden_states=True)
        if hasattr(conf, 'num_concepts'): # a model with latent conecpts as outputs
            trust_remote_code = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=transformer!='EMBEDDIA/sloberta')
        if mlm:
            model = AutoModelForMaskedLM.from_pretrained(transformer, config=conf, trust_remote_code=trust_remote_code)
        else:
            model = AutoModel.from_pretrained(transformer, config=conf)
        model.to(self.device)
        return tokenizer, model

    def read_sequences(self, in_file, limit=-1):
        raise Exception("Method unimplemented")

    def read_sequences_with_embeddings(self, in_file, limit=-1, average=False, batch_size=512):
        if self.transformer_model is None:
            raise Exception('Reader object was not initialized with a transformer model')

        is_tagged_list, sequence_list = [], []
        for i,(sequence, is_tagged) in enumerate(self.read_sequences(in_file, limit)):
            is_tagged = [True] if average else is_tagged
            is_tagged_list.append(is_tagged)
            sequence_list.append(sequence)
            if len(sequence_list) == batch_size:
                for a,b,c in zip(self.process_sequence_new(sequence_list, average), sequence_list, is_tagged_list):
                    yield a,b,c
                is_tagged_list, sequence_list = [], []
        if len(sequence_list) > 0:
            for a,b,c in zip(self.process_sequence_new(sequence_list, average), sequence_list, is_tagged_list):
                yield a,b,c

    def process_sequence_new(self, sequences, average):
        its, tokens = [], []
        #tokens_sanities = []
        for seq in sequences:
            #tokens_sanity, indexed_tokens_sanity = self.tokenize_sequence(seq)
            #tokens_sanities.append(tokens_sanity)
            its.append(self.tokenizer(seq, is_split_into_words=True, return_token_type_ids=False))
            tokens.append(its[-1].word_ids())
        indexed_tokens = self.collator(its).to(self.device)

        vecs = self.get_vecs(indexed_tokens)
        for seq_id in range(len(sequences)):
            seq_vecs = [vecs[layer_id][seq_id] for layer_id in range(len(vecs))]
            tok_ids = self.convert_token_ids(tokens[seq_id])
            #sanity_check = np.all([a==b for a,b in zip(tok_ids, tokens_sanities[seq_id])])
            #if sanity_check == False:
            #    print(tok_ids, tokens_sanities[seq_id], seq_id)
            yield self.get_representation_new(tok_ids, seq_vecs, average)
    
    def convert_token_ids(self, token_membership):
        to_return = []
        current_id = -1
        for i, tm in enumerate(token_membership):
            if tm is not None and tm != current_id:
                to_return.append(i)
                current_id = tm
        to_return.append(i)
        return to_return

    def get_vecs(self, indexed_tokens):
        vecs = None
        with torch.no_grad():
            output = self.model(**indexed_tokens.to(self.device))
            if self.mlm:
                extra_symbols = getattr(self.model.config, 'num_concepts', 0)
                vecs = torch.nn.functional.softmax(output['logits'][:,:,-extra_symbols:], dim=-1)
            else:
                vecs = list(output['hidden_states'])
                if hasattr(self.model, 'final_norm'):
                    vecs.append(self.model.final_norm(vecs[-1]))
        return vecs

    def get_representation_new(self, word_mapping, vecs, average):
        per_layer_embeddings = []
        for emb in vecs:
            if self.mlm:
                emb = emb.unsqueeze(0)
            if average: # XXX TODO test this 
                averaged = torch.mean(emb, dim=0).detach().cpu().numpy().reshape(1,-1)
                per_layer_embeddings.append(averaged)
            else:
                token_embeddings = []
                for k,l in zip(word_mapping, word_mapping[1:]):
                    ki, li = k, l
                    if self.pooling_strategy == 'first':
                        li=ki+1
                    elif self.pooling_strategy == 'last':
                        ki=li-1
                    elif self.pooling_strategy == 'norm':
                        norms = torch.linalg.norm(emb[k:l], dim=1)
                        ki += torch.argmax(norms).item()
                        li = ki + 1
                    token_embeddings.append(torch.mean(emb[ki:li], dim=0).detach().cpu().numpy())
                per_layer_embeddings.append(np.array(token_embeddings))
        return per_layer_embeddings
    
    def tokenize_sequence(self, sequence):
        orig_to_tok_map, transformer_tokens = [], []
        for tok_pos, orig_token in enumerate(sequence):
            orig_to_tok_map.append(len(transformer_tokens))
            transformer_tokens.extend(self.tokenizer.tokenize('{}{}'.format(' ' if tok_pos>0 else '', orig_token)))

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
            output = self.model(torch.tensor([indexed_tokens_with_specials]).to(self.device))
            if self.mlm:
                extra_symbols = getattr(self.model.config, 'num_concepts', 0)
                vecs = torch.nn.functional.softmax(output['logits'][:,:,-extra_symbols:], dim=-1)
            else:
                vecs = list(output['hidden_states'])
                if hasattr(self.model, 'final_norm'):
                    vecs.append(self.model.final_norm(vecs[-1]))

        per_layer_embeddings = []
        for emb in vecs:
            if self.mlm:
                emb = emb.unsqueeze(0)
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

    def read_sequences(self, in_file, limit=-1, annotated=False):
        root = ET.parse(in_file).getroot()
        for i,s in enumerate(root.findall('text/sentence')):
            if i==limit: break

            seq_tokens, is_tagged, metadata, ids = [], [], [], []
            for orig_token in list(s):
                seq_tokens.append(orig_token.text)
                is_tagged.append(orig_token.tag=='instance')
                normalized_pos = 'r'
                if len(orig_token.attrib['pos'])>0 and orig_token.attrib['pos']!="ADV": normalized_pos = orig_token.attrib['pos'][0].lower()
                metadata.append(f'{orig_token.attrib['lemma']}.{normalized_pos}')
                ids.append(orig_token.get('id', ''))
            if self.mask:
                mask_positions = np.where(is_tagged)[0]
                for mp in mask_positions:
                    new_seq_tokens, new_is_tagged = [], []
                    for i,st in enumerate(seq_tokens):
                        new_seq_tokens.append(st if i != mp else self.tokenizer.mask_token)
                        new_is_tagged.append(i==mp)
                    yield new_seq_tokens, new_is_tagged
            else:
                if annotated:
                    yield seq_tokens, is_tagged, metadata, ids
                else:
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
                lemma = f'{token.get('lemma', token.text)}{pos_delim}{normalized_pos}'
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

    def get_tokens(self, in_file, pwn=None):
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

    def get_tokens(self, in_file=None, pwn=None):
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


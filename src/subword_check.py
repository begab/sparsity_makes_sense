import os, sys

from utils.readers import *
from transformers import AutoTokenizer

import argparse
import torch
import transformers
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class Preprocessor(object):

    def __init__(self, dataset_reader, transformer, gpu=-1, mlm=True):
        klass = globals()[dataset_reader]
        self.mlm = mlm
        self.reader = klass(transformer, gpu, mlm=self.mlm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determines contextualized representation for sequences in a flexible manner.')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--reader')
    parser.add_argument('--in_files', nargs='+')

    args = parser.parse_args()

    p = Preprocessor(args.reader, args.transformer, args.gpu_id, mlm=True)
    for i, in_file in enumerate(args.in_files):
        total_mlm_loss = {layer:0 for layer in range(-4,0)}
        total_corrects = {l:0 for l in total_mlm_loss}
        total_masked_tokens = 0
        np.random.seed(42)

        fertility, multi_subwords, is_tagged = [], [], []
        for ii, (sequence, is_labeled) in enumerate(p.reader.read_sequences(in_file)):   
            is_tagged.extend(is_labeled)
            tokens, input_ids = p.reader.tokenize_sequence(sequence)
            #logging.info(sequence)
            for token, token_from, token_end, it in zip(sequence, tokens, tokens[1:], is_tagged):
                #logging.info("{}\t{}\t{}".format(token_from,token_end,token_end-token_from))
                #print('{}\t{}\t{}\t{}\t{}'.format(args.transformer, token_end - token_from, it, token, in_file))
                fertility.append(token_end - token_from)
                multi_subwords.append(1 if token_end - token_from > 1 else 0)
            #logging.info((len(sequence), len(tokens), len(is_labeled), len(indexed_tokens)))
        
            if p.mlm:
                with torch.no_grad():
                  input_ids = torch.tensor([input_ids]).to(p.reader.device)
                  non_special_subwords = input_ids.shape[1] - 2
                  ids_to_mask = sorted(1 + np.random.choice(non_special_subwords, int(0.15*non_special_subwords), replace=False))
                  if len(ids_to_mask)>0:
                      total_masked_tokens += len(ids_to_mask)
                      masked_tokens = input_ids[0][ids_to_mask]

                      labels = -100*torch.ones_like(input_ids)
                      for position in ids_to_mask:
                          labels[0][position] = input_ids[0][position]
                      input_ids[0][ids_to_mask] = p.reader.tokenizer.mask_token_id

                      res = p.reader.model(input_ids=input_ids, labels=labels)
                      loss_fun = torch.nn.CrossEntropyLoss()
                      for layer in range(-4, 0):
                          if isinstance(p.reader.model, transformers.models.roberta.RobertaForMaskedLM):
                              lm_head = p.reader.model.lm_head
                          elif isinstance(p.reader.model, transformers.models.electra.ElectraForMaskedLM):
                              lm_head = p.reader.model.generator_lm_head
                          else:
                              lm_head = p.reader.model.cls
                          logits_at_layer = lm_head(res.hidden_states[layer])
                          vocab_size = logits_at_layer.shape[-1]
                          l = loss_fun(logits_at_layer.view(-1, vocab_size), labels.view(-1))
                          total_mlm_loss[layer] += l.item() * len(ids_to_mask)
                          total_corrects[layer] += (masked_tokens==logits_at_layer[labels!=-100].argmax(axis=1)).sum().item()

        mean_fertility = np.mean(fertility)
        multi_token_ratio = np.sum(multi_subwords) / len(is_tagged)
        mean_fertility_labeled = np.mean([f for f,t in zip(fertility, is_tagged) if t])
        multi_token_ratio_labeled = np.sum([m for m,t in zip(multi_subwords, is_tagged) if t]) / np.sum(is_tagged)

        print('fertility\t{}\t{}\t{}'.format(mean_fertility, args.transformer, in_file))
        print('MTR\t{}\t{}\t{}'.format(multi_token_ratio, args.transformer, in_file))
        print('fertility-L\t{}\t{}\t{}'.format(mean_fertility_labeled, args.transformer, in_file))
        print('MTR-L\t{}\t{}\t{}'.format(multi_token_ratio_labeled, args.transformer, in_file))
        for layer in total_mlm_loss:
            print('MLM\t{}\t{}\t{}\t{}'.format(total_mlm_loss[layer] / total_masked_tokens, args.transformer, in_file, layer))
            print('Acc\t{}\t{}\t{}\t{}'.format(total_corrects[layer] / total_masked_tokens, args.transformer, in_file, layer))
        
        #logging.info('{}\t{}\t{}\t{}\t{}'.format(ii, args.transformer, len(is_tagged), len(fertility), np.sum(is_tagged)))
        #logging.info('{}\t{}\t{:.3f}'.format(args.transformer, in_file, mean_fertility))
        #logging.info('{}\t{}\t{:.3f}'.format(args.transformer, in_file, multi_token_ratio))
        #logging.info('{}\t{}\t{:.3f}'.format(args.transformer, in_file, mean_fertility_labeled))
        #logging.info('{}\t{}\t{:.3f}'.format(args.transformer, in_file, multi_token_ratio_labeled))

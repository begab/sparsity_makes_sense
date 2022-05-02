import os, sys
from utils.readers import *
from collections import defaultdict
from utils.cka import feature_space_linear_cka

from sklearn.metrics.pairwise import cosine_distances
from utils.utils import row_normalize
import torch

import numpy as np
import datasets
import word2word
from konlpy.tag import Mecab

import pickle
import argparse
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s\t%(asctime)s\t%(levelname)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

def getknn(sc, x, y, k=10):
    if type(sc) == torch.tensor:
        top_vals, sidx = sc.topk(top_k, dim=1, largest=True)
        f = top_vals.sum()
        ytopk = y[sidx.flatten(), :].reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
        df = torch.sum(ytopk, dim=1).T @ x
    else:
        sidx = np.argpartition(sc, -k, axis=1)[:, -k:]
        f = np.sum(sc[np.arange(sc.shape[0])[:, None], sidx])
        ytopk = y[sidx.flatten(), :].reshape(sidx.shape[0], sidx.shape[1], y.shape[1])
        df = np.dot(ytopk.sum(1).T, x)
    return f / k, df / k

def rcsls(tgt, src, Z_tgt, Z_src, R, knn=10):
    tgt_modded = tgt @ R.T
    f = 2 * (tgt_modded * src).sum()
    df = 2 * src.T @ tgt
    fk0, dfk0 = getknn(tgt_modded @ Z_src.T, tgt, Z_src, knn)
    fk1, dfk1 = getknn(((Z_tgt @ R.T) @ src.T).T, src, Z_tgt, knn)
    f = f - fk0 - fk1
    df = df - dfk0 - dfk1.T
    return -f / tgt.shape[0], -df / tgt.shape[0]

def calculate_rcsls(R, tgt, src, spectral, batchsize=0, niter=10, knn=10, maxneg=50000, lr=1.0, verbose=False):
    np.random.seed(400)
    fold, Rold = 0, []
    for it in range(0, niter+1):
        if lr < 1e-4:
            break

        indices = list(range(tgt.shape[0]))
        if len(indices) > batchsize > 0:
            indices = np.random.choice(tgt.shape[0], size=batchsize, replace=False)
        f, df = rcsls(tgt[indices, :], src[indices, :], tgt[:maxneg, :], src[:maxneg, :], R, knn)
        R -= lr * df
        if spectral:
            U, s, V = np.linalg.svd(R)
            s[s > 1] = 1
            s[s < 0] = 0
            R = U @ (np.diag(s) @ V)
        if verbose:
            logging.info("[it={}] (lr={:.4f}) f = {:.4f}".format(it, lr, f))

        if f > fold and it > 0 and batchsize<=0:
            lr /= 2
            f, R = fold, Rold
        fold, Rold = f, R
    return f, R.T


def main():

    parser = argparse.ArgumentParser(description='Performs WSD model training.')

    parser.add_argument('--src_transformer', default='bert-large-cased')
    parser.add_argument('--tgt_transformer')
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--tgt_lang')
    #parser.add_argument('--src_layer', type=int)
    #parser.add_argument('--tgt_layer', type=int)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--src_pooling', default='mean', choices='mean-first-last-norm'.split('-'))
    parser.add_argument('--tgt_pooling', default='mean', choices='mean-first-last-norm'.split('-'))
    parser.add_argument('--ko_tokenizer_dir', default='/home/anonymous/.conda/envs/sms/lib/mecab/dic/mecab-ko-dic/')

    parser.add_argument('--center', dest='center', action='store_true')
    parser.add_argument('--not-center', dest='center', action='store_false')
    parser.set_defaults(center=False)

    parser.add_argument('--use-pytorch', dest='pt', action='store_true')
    parser.add_argument('--not-use-pytorch', dest='pt', action='store_false')
    parser.set_defaults(pt=False)
    
    args = parser.parse_args()

    #logging.info(args)
    out_dir_name = os.path.dirname(args.out_dir)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)

    out_file = '{}/{}_{}/{}_{}.pkl'.format(args.out_dir, args.src_pooling, args.tgt_pooling, args.src_transformer.replace('/','_'), args.tgt_transformer.replace('/','_'))

    threshold = 25000
    if os.path.exists(out_file):
        X,Y,x,y = pickle.load(open(out_file, 'rb'))
    else:
        readers = {args.src_lang: SeqReader(args.src_transformer, np.abs(args.gpu_id), args.src_pooling),
                   args.tgt_lang: SeqReader(args.tgt_transformer, np.abs(args.gpu_id), args.tgt_pooling)}

        tatoeba_src_lang = args.src_lang if args.src_lang != 'zh' else 'cmn'
        tatoeba_tgt_lang = args.tgt_lang if args.tgt_lang != 'zh' else 'cmn'
        w2w_src_lang = args.src_lang if args.src_lang != 'zh' else 'zh_cn'
        w2w_tgt_lang = args.tgt_lang if args.tgt_lang != 'zh' else 'zh_cn'
        tatoeba = datasets.load_dataset("tatoeba", lang1=min(tatoeba_src_lang, tatoeba_tgt_lang), lang2=max(tatoeba_src_lang, tatoeba_tgt_lang))
        number_of_translated_sentences = len(tatoeba['train'])

        tokenizers = {}
        if args.src_lang == 'ko':
             tokenizers['ko'] = Mecab(args.ko_tokenizer_dir)
        else:
            tokenizers[args.src_lang] = word2word.tokenization.load_tokenizer(w2w_src_lang)
        if args.tgt_lang == 'ko':
            tokenizers['ko'] = Mecab(args.ko_tokenizer_dir)
        else:
            tokenizers[args.tgt_lang] = word2word.tokenization.load_tokenizer(w2w_tgt_lang)

        w2w = word2word.Word2word(w2w_src_lang, w2w_tgt_lang)
        w2w2 = word2word.Word2word(w2w_tgt_lang, w2w_src_lang)

        X, Y = defaultdict(list), defaultdict(list)
        x, y = [], []
        for i, tr in enumerate(tatoeba['train']['translation']):
            if i%2500==0 or len(x)>threshold:
                logging.info('{}\t{}\t{}\t{}'.format(i, number_of_translated_sentences, tr, len(x)))
                if len(x)>threshold: break
            if np.product([len(s.strip()) for s in tr.values()]) == 0: continue # both sentences has to be non-empty
            token_mappings = {l:{} for l in readers}
            vecs = {l:{} for l in readers}
            for (lang, sent) in tr.items():
                lang = 'zh' if lang=='cmn' else lang
                sent_toks = word2word.tokenization.word_segment(sent, lang if lang!='zh' else 'zh_cn', tokenizers[lang])
                orig_to_tok_map, indexed_tokens_with_specials = readers[lang].tokenize_sequence(sent_toks)
                token_mappings[lang] = {t:i for i,t in enumerate(sent_toks)}
                vecs[lang] = readers[lang].get_representation(orig_to_tok_map, indexed_tokens_with_specials, False)
            for tok, pos in token_mappings[args.src_lang].items():
                if tok in w2w.word2x:
                    for translation in w2w(tok):
                      condition1 = translation in token_mappings[args.tgt_lang]
                      condition2 = translation in w2w2.word2x and tok in set(w2w2(translation))
                      if condition1 and condition2:
                          for layer in range(len(vecs[args.src_lang])):
                              X[layer].append(vecs[args.src_lang][layer][pos])
                          for layer in range(len(vecs[args.tgt_lang])):
                              Y[layer].append(vecs[args.tgt_lang][layer][token_mappings[args.tgt_lang][translation]])
                          x.append(tok)
                          y.append(translation)
    
        #with open(out_file, 'wb') as fo:
        #    pickle.dump((X,Y,x,y), fo)

    for layer in X:
        X[layer] = X[layer][:threshold]
    for layer in Y:
        Y[layer] = Y[layer][:threshold]
    x,y = x[:threshold], y[:threshold]
    np.random.seed(42)
    permutation = np.random.permutation(range(len(x)))
    x = [x[p] for p in permutation]
    y = [y[p] for p in permutation]
    limit = int(0.8 * len(x))
    for l1, X_emb in X.items():
        for l2, Y_emb, in Y.items():
            if l1 < len(X) - 4 or l2 < len(Y) - 4: continue
            tgt = np.hstack([np.array(Y_emb),
                             np.zeros((len(Y_emb),  X_emb[0].shape[0] - Y_emb[0].shape[0]))])
            src = np.array(X_emb)
            if args.center:
                tgt -= np.mean(tgt[:limit], axis=0)
                src -= np.mean(src[:limit], axis=0)
            tgt = row_normalize(tgt)
            src = row_normalize(src)
            tgt = np.array([tgt[p] for p in permutation])
            src = np.array([src[p] for p in permutation])

            if args.pt == True:
                src = torch.tensor(src).to('cuda:{}'.format(args.gpu_id))
                tgt = torch.tensor(tgt).to('cuda:{}'.format(args.gpu_id))
                U, _, V = torch.linalg.svd(src[0:limit].T@tgt[0:limit])
            else:            
                U, _, V = np.linalg.svd(src[0:limit].T@tgt[0:limit])

            isometric_trafo = V.T @ U.T
            identity_trafo = np.eye(isometric_trafo.shape[0], isometric_trafo.shape[1]) 
            f, rcsls_trafo = calculate_rcsls(isometric_trafo.T, tgt, src, False, niter=10, verbose=False)

            if args.pt == True:
                src, tgt = src.cpu().numpy(), tgt.cpu().numpy()

            for method, trafo in zip(['isometric', 'rcsls', 'identity'], [isometric_trafo, rcsls_trafo, identity_trafo]):
                if trafo is None: continue
                if trafo == torch.tensor:
                    trafo = trafo.cpu().numpy()
 
                trafo_file = '{}{}/{}-{}/{}_{}_{}_{}_{}_{}'.format(args.out_dir, '-pt' if args.pt else '', args.src_pooling, args.tgt_pooling, method, args.src_transformer.replace('/','_'), l1, args.tgt_transformer.replace('/','_'), l2, limit)
                if method != 'identity':
                    np.save(trafo_file, trafo)
                cka = feature_space_linear_cka(src[limit:], (tgt @ trafo)[limit:])
                sims = cosine_distances(src[limit:], (tgt @ trafo)[limit:])
                s = np.argsort(sims, axis=1)
                avg_sims = np.mean(sims)
                for topk in [1, 5, 10, 15]:
                    accuracy = sum([1 if i in set(j) else 0 for i,j in enumerate(s[:,0:topk])]) / (len(x)-limit)
                    # accuracy2 is more permissive as returning any sentence with the etalon translation is counted as a correct mapping
                    accuracy2 = sum([1 if x[i] in set([x[jj] for jj in j]) else 0 for i,j in enumerate(s[:,0:topk])])/(len(x)-limit)
                    top_sims_avg = np.mean([sims[row, s[row, -top-1]] for row in range(sims.shape[0]) for top in range(topk)])
                    logging.info("{}-{}-{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(method, args.src_pooling, args.tgt_pooling, args.src_lang, args.tgt_lang, args.src_transformer, args.tgt_transformer, l1, l2, limit, len(x) - limit, topk, accuracy, accuracy2, top_sims_avg, avg_sims, cka))

if __name__ == '__main__':
    main()

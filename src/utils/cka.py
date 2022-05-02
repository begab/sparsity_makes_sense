'''
The original implementation of CKA from https://github.com/google-research/google-research/tree/master/representation_similarity
'''

import numpy as np
from collections import defaultdict, Counter
from .readers import SemcorReader

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)

if __name__=='__main__':
    #L1=np.load('/data/berend/representations_unreduced/bert-large-cased/semcor.data.xml_bert-large-cased_avg_False_layer_1.npy')
    #L21=np.load('/data/berend/representations_unreduced/bert-large-cased/semcor.data.xml_bert-large-cased_avg_False_layer_21.npy')

    sr = SemcorReader()
    Y, rows_with_annotation = [], []
    y_to_id = {}
    lemmas = defaultdict(list)
    for i,t in enumerate(sr.get_tokens("/data/berend/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml")):
        if t[2] is not None:
            Y.append([])
            rows_with_annotation.append(i)
            for synset in t[0]:
                sid = len(y_to_id)
                if synset in y_to_id:
                    sid = y_to_id[synset]
                y_to_id[synset] = sid
                Y[-1].append(sid)
                lemmas[t[3]].append((synset, i))
    Y_matrix = np.zeros((len(Y), len(y_to_id)))
    for row,y in enumerate(Y):
        for synset_id in y:
            Y_matrix[row, synset_id] = 1
    label_filter = Y_matrix.sum(axis=0) >= 5
    Y_matrix = Y_matrix[:, label_filter]
    row_filter = Y_matrix.sum(axis=1)>0
    Y_matrix = Y_matrix[row_filter]

    print(Y_matrix.shape, len(Y), len(y_to_id), len(rows_with_annotation))
    print(Y[0:10])
    print(rows_with_annotation[0:10])
    print([(y,y_to_id[y]) for y in list(y_to_id.keys())[0:10]])

    candidates = {}
    for i,l in enumerate(open('/data/berend/WSD_Evaluation_Framework/Data_Validation/candidatesWN30.txt')):
        lemma, pos, *senses = l.strip().split()
        lp = '{}.{}'.format(lemma, pos)
        #if lp in entropies:
        candidates[lp] = len(senses)

    entropies, ckas = {}, {}
    for l1 in range(0, 25):
        L1 = np.load('/data/berend/representations_unreduced/bert-large-cased/semcor.data.xml_bert-large-cased_avg_False_layer_{}.npy'.format(l1))
        layerwise_cka_score = feature_space_linear_cka(L1[rows_with_annotation][row_filter], Y_matrix)
        print(l1, layerwise_cka_score)
        continue
        for l2 in range(l1+1, 25):
            L2 = np.load('/data/berend/representations_unreduced/bert-large-cased/semcor.data.xml_bert-large-cased_avg_False_layer_{}.npy'.format(l2))
            for i, (lemma, occurrances) in enumerate(lemmas.items()):
                c = Counter([o[0] for o in occurrances])
                total = sum(c.values())
                if total > 50 and len(c.values()) > 1:
                    if lemma not in entropies:
                        entropy = -np.sum([v/total * np.log2(v/total) for v in c.values()])
                        normalized_entropy = entropy / np.log2(len(c.values()))
                        indices = [o[1] for o in occurrances]
                        #print(lemma, c, total, entropy, cka_from_features)
                        entropies[lemma] = normalized_entropy
                        ckas[lemma] = {}
                    cka_from_features = feature_space_linear_cka(L1[indices], L2[indices])
                    ckas[lemma][(l1,l2)] = cka_from_features
            print(l1, l2)

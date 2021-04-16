from argparse import ArgumentParser
from collections import Counter
from pprint import pprint
import os


def get_pos_from_key(key):
    """
    assumes key is in the wordnet key format, i.e., 's_gravenhage%1:15:00:
    :param key: wordnet key
    :return: pos tag corresponding to the key
    """
    numpos = key.split("%")[-1][0]
    if numpos == "1":
        return "NOUN"
    elif numpos == "2":
        return "VERB"
    elif numpos == "3" or numpos == "5":
        return "ADJ"
    else:
        return "ADV"


def parse_file(path):
    id2ans = dict()
    with open(path) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            id, *answers = fields
            id2ans[id] = set(answers)
    return id2ans


def get_extended_pos(char_pos):
    return "NOUN" if char_pos == "n" else "VERB" if char_pos == "v" \
        else "ADJ" if char_pos == "a" else "ADV"


def get_pos(label):
    if label.startswith("bn:") or label.startswith("wn:"):
        return get_extended_pos(label[-1])
    return get_pos_from_key(label)


def get_bn_labels(labels, wnkey2bn):
    bn_labels = set()
    for l in labels:
        if l.startswith("bn:"):
            bn_labels.add(l)
            continue
        else:
            bn_labels.add(wnkey2bn[l])
    return bn_labels


def evaluate(answers, golds, by_pos, wnkey2bn=None):
    correct = 0
    tot = 0
    correct_by_pos = Counter()
    tot_by_pos = Counter()
    for id in golds.keys():
        ans = answers[id]
        labels = golds[id]
        if wnkey2bn is not None:
            labels = get_bn_labels(labels, wnkey2bn)
        pos = get_pos(list(labels)[0])

        if len(ans & labels) > 0:
            correct += 1
            correct_by_pos[pos] += 1

        tot += 1
        tot_by_pos[pos] += 1
    accuracy = correct / tot
    results = {}
    if by_pos:
        for p in "NOUN,VERB,ADJ,ADV".split(","):
            p_tot = tot_by_pos[p]
            p_cor = correct_by_pos[p]
            if p_tot > 0:
                results[p] = p_cor / p_tot

    results["ALL"] = accuracy
    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answer_file", required=True)
    parser.add_argument("--gold_file", required=True)
    parser.add_argument("--by_pos", action="store_true", default=False)

    args = parser.parse_args()

    answer_file = args.answer_file
    gold_file = args.gold_file
    by_pos = args.by_pos

    id2answer = parse_file(answer_file)
    id2gold = parse_file(gold_file)
    results = evaluate(id2answer, id2gold, by_pos)
    for k, v in results.items():
        print(k, v)

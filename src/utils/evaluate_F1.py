from argparse import ArgumentParser
from collections import defaultdict

def load_keys(path):
    keys = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            instance_id, *sense_keys = line.split()
            if not sense_keys:
                continue
            assert instance_id not in keys
            keys[instance_id] = set(sense_keys)
    
    return keys


def evaluate_macroF1(gold, pred, eval_keys=None, strict=False):
    tp = defaultdict(lambda: 0.)
    fp = defaultdict(lambda: 0.)
    fn = defaultdict(lambda: 0.)
    gold_keys = set()

    for instance_id in gold:
        instance_gold = gold[instance_id]
        
        if eval_keys and instance_id not in eval_keys:
            continue

        instance_tp, instance_fp = 0., 0.
        num_instance_predictions = 1
        
        if instance_id in pred:
            instance_predictions = pred[instance_id]
            num_instance_predictions = len(instance_predictions)

            for key in instance_predictions:
                if key in instance_gold:
                    instance_tp = 1. / num_instance_predictions
                else:
                    instance_fp = 1. / num_instance_predictions
            
            for key in instance_predictions:
                fp[key] += instance_fp
        
        for key in instance_gold:
            gold_keys.add(key)
            tp[key] += instance_tp
            if strict:
                if key not in instance_predictions:
                    fn[key] += 1. / num_instance_predictions
            else:
                if instance_tp == 0.:
                    fn[key] += 1. / num_instance_predictions

    avg_p = 0.
    avg_r = 0.
    avg_f1 = 0.
    total = 0

    for key in gold_keys:
        key_tp = tp[key] if key in tp else 0
        key_fp = fp[key] if key in fp else 0
        key_fn = fn[key] if key in fn else 0
        if key_tp == 0 and key_fp == 0 and key_fn == 0:
            continue

        p = key_tp / (key_tp + key_fp) if key_tp + key_fp != 0 else 0
        r = key_tp / (key_tp + key_fn) if key_tp + key_fn != 0 else 0
        f1 = 2 * (p * r) / (p + r) if p + r != 0 else 0

        avg_p += p
        avg_r += r
        avg_f1 += f1
        total += 1
    
    avg_p /= total
    avg_r /= total
    avg_f1 /= total
    
    #print('Macro Precision =', '{:0.2f}'.format(100. * avg_p))
    #print('Macro Recall    =', '{:0.2f}'.format(100. * avg_r))
    #print('Macro F1 score  =', '{:0.2f}'.format(100. * avg_f1))
    return avg_f1

def evaluate_microF1(gold, pred, eval_keys=None):
    tp, fp, fn = 0, 0, 0

    for instance_id in gold:
        if eval_keys and instance_id not in eval_keys:
            continue

        if instance_id in pred:
            correct = False

            for key in pred[instance_id]:
                if key in gold[instance_id]:
                    correct = True
            
            if correct:
                tp += 1
            else:
                for key in pred[instance_id]:
                    fp += 1
                fn += 1
        else:
            fn += 1
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    #print('Precision   =', '{:0.2f} [{}/{}]'.format(100 * precision, tp, tp + fp))
    #print('Recall      =', '{:0.2f} [{}/{}]'.format(100 * recall, tp, tp + fn))
    #print('F1 score    =', '{:0.2f}'.format(100 * f1))
    return f1

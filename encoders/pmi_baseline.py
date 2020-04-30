import itertools
from collections import Counter, defaultdict
import numpy as np
from encoders.model_utils import load_json, save_json, p_r_f, print_result_summary, get_auc
import csv
import configparser
import argparse
"""
this baseline computes a hashtag2class dictionary using pmi scores of the training data
"""

def compute_pmis(seqs, labels, labelset):
    token_counts = Counter(itertools.chain.from_iterable([seq.split() for seq in seqs]))
    label_counts = Counter(labels)
    token_sum = np.sum(list(token_counts.values()))
    print(token_sum)
    label_sum = np.sum(list(label_counts.values()))
    coocs = {}
    for seq, label in zip(seqs, labels):
        for tok in seq.split():
            coocs.setdefault(tok, defaultdict(int))[label] += 1
    pmis = {}
    for tok, tok_count in token_counts.items():
        for l, label in enumerate(labelset):
            p_x_y = coocs[tok][label]/float(label_counts[label])
            if p_x_y == 0:
                pmi = 0
            else:
                p_x = tok_count/float(token_sum)
                pmi = np.log(p_x_y/p_x)
            pmis.setdefault(tok, []).append(pmi)
    return pmis, coocs


def sents2seqs(sents):
    train_seqs = []
    for sent in sents:
        train_seqs.append(' '.join([elm for elm in sent.split() if elm.startswith('#') and elm != '#mh17']))
    return train_seqs


def majority(labels):
    c = Counter(labels)
    if len(set(c.values())) == 1:
        return np.random.choice(labels)
    else:
        return c.most_common(1)[0][0]

def predict(seqs, pmis, coocs, thr=3):
    oovs = []
    multis = []
    predictions = []
    probs = []
    for seq in seqs:
        seq_predictions = []
        for elm in seq.split():
            # check that the hashtag occurs more than threshold times

            if elm in pmis and sum(list(coocs[elm].values())) > thr:
                print(sum(list(coocs[elm].values())))
                pred = labelset[np.argmax(pmis[elm])]
                seq_predictions.append(pred)

        if len(seq_predictions) == 0:
            oovs.extend(seq.split())
            predictions.append(np.random.choice(labelset))
        elif len(seq_predictions) > 1:
            # check if we can make a majority decision
            majority_label = majority(seq_predictions)
            predictions.append(majority_label)
            if len(set(seq_predictions)) > 1:
                multis.append(len(set(seq_predictions)))
        else:
            predictions.append(seq_predictions[0])

    return predictions, oovs, multis


def predict_random(seqs, labelset):
    preds = []
    for seq in seqs:
        preds.append(np.random.choice(labelset))
    return preds


def write_results_and_hyperparams(fname, results, params, labelset):

    metrics = ['p', 'r', 'f']
    results_prefixed = {}
    for key in ['macro', 'micro']:
        for i, m in enumerate(metrics):
            results_prefixed['{}_{}'.format(key, metrics[i])] = results[key][i]
    for label, val in results['aucs'].items():
        results_prefixed['{}_auc'.format(label)] = val
    results_prefixed['macro_auc'] = np.mean(list(results['aucs'].values()))
    for key in results['per_class'].keys():
        for i, m in enumerate(metrics):
            results_prefixed['{}_{}'.format(key, metrics[i])] = results['per_class'][key][i]

    # add cm results
    for i, label_i in enumerate(labelset):
        for j, label_j in enumerate(labelset):
            results_prefixed['cm_{}{}'.format(label_i, label_j)] = results['cm'][i][j]

    if 'best_epoch' in results:
        results_prefixed['best_epoch'] = results['best_epoch']
    if 'best_macro' in results:
        results_prefixed['best_macro_f'] = results['best_macro_f']

    with open(fname, 'r', encoding='utf-8') as csvfile:
        header_reader = csv.reader(itertools.islice(csvfile, 0, 1), delimiter=',', quotechar='"')
        header = [elm for elm in header_reader][0]
    csvfile.close()

    results_prefixed.update({key: val for key, val in params.items() if key in header})
    values = results_prefixed

    for elm in header:
        if elm not in values:
            values[elm] = 'Missing'

    with open(fname, 'a', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', quotechar='"', fieldnames=header)
        writer.writerow(values)
    csvfile.close()




def main(args):

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    np.random.seed(0)
    auc1 = []
    auc2 = []
    auc3 = []
    auc_macro = []
    f1 = []
    f2 = []
    f3 = []
    f_macro = []
    for split in range(10):
        d= load_json(config.get('Files', 'data_{}'.format(split)))
        train_seqs = sents2seqs(d['train']['seq'])
        train_labels = d['train']['label']


        labelset = list(set(train_labels))
        pmis, coocs = compute_pmis(train_seqs, train_labels, labelset)

        sorted_keys = sorted(list(pmis.keys()), key=lambda x: np.max(pmis[x]), reverse=True)

        for tok in sorted_keys:
            vals = pmis[tok]
            print(tok, vals, [coocs[tok][label] for label in labelset])

        test_seqs = sents2seqs(d['test']['seq'])
        test_labels = d['test']['label']


        test_preds = predict_random(test_seqs, labelset)
        test_probs = []
        for pred in test_preds:
            test_probs.append([1 if pred == elm else 0 for elm in labelset ])

        label2idx = {label: labelset.index(label) for label in labelset}
        results = p_r_f([label2idx[label] for label in test_labels], [label2idx[label] for label in test_preds], labelset)
        aucs, precs, recs, thr = get_auc([label2idx[label] for label in test_labels], test_probs, labelset)
        results['aucs'] = aucs


        print(print_result_summary(results))
        print(results['aucs'])

        f1.append(results['per_class']['1'][2])
        f2.append(results['per_class']['2'][2])
        f3.append(results['per_class']['3'][2])
        f_macro.append(results['macro'][2])
        auc1.append(results['aucs']['1'])
        auc2.append(results['aucs']['2'])
        auc3.append(results['aucs']['3'])
        auc_macro.append(np.mean([results['aucs'][label] for label in ['1', '2', '3']]))


    print(len(f1))
    print('f1: {}'.format(np.mean(f1)))
    print('f2: {}'.format(np.mean(f2)))
    print('f3: {}'.format(np.mean(f3)))
    print('f_macro: {}'.format(np.mean(f_macro)))

    print('auc1: {}'.format(np.mean(auc1)))
    print('auc2: {}'.format(np.mean(auc2)))
    print('auc3: {}'.format(np.mean(auc3)))
    print('auc_macro: {}'.format(np.mean(auc_macro)))


if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description='Tweet classification PMI baseline')
    parser.add_argument('--config', type=str,
                        default='../config.cfg',
                        help="Config file")
    args = parser.parse_args()
    main(args)
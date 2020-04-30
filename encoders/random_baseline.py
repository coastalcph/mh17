import numpy as np
from encoders.model_utils import load_json, p_r_f, print_result_summary
import argparse
import configparser


def main(args):

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)
    np.random.seed(42)
    d = load_json(config.get('Files', 'data'))

    train_labels = d['train']['label']
    labelset = list(set(train_labels))

    dev_labels = d['dev']['label']

    preds = []
    for elm in dev_labels:
        preds.append(np.random.choice(labelset))

    label2idx = {label: labelset.index(label) for label in labelset}
    results = p_r_f([label2idx[label] for label in dev_labels], [label2idx[label] for label in preds], labelset)
    print(print_result_summary(results))


if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description='Tweet classification with random baseline')
    parser.add_argument('--config', type=str,
                        default='../config.cfg',
                        help="Config file")
    args = parser.parse_args()
    main(args)

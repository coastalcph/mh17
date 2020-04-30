import torch.nn as nn
import torch.optim as optim

import configparser

from encoders import param_reader
from encoders.model_utils import *
from encoders.model_utils import p_r_f, print_result_summary, log_params


from encoders import param_reader
from encoders.model_utils import *
from encoders.model_utils import p_r_f, print_result_summary, log_params
from sklearn.metrics import confusion_matrix
import uuid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import  SVC
import numpy as np
import uuid

UNK = 'UNK'

def evaluate_validation_set(model, seqs, golds, lengths, sentences, labelset, compute_auc=False):

    probs = model.predict_proba(seqs)
    y_true = golds
    y_pred = np.argmax(probs,1)
    loss = 0
    results = p_r_f(y_true, y_pred, labelset)
    cm = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(labelset))])
    results['cm'] = cm
    if compute_auc is True:
        y_true = [int(elm) for elm in y_true]
        aucs, precs, recs, thr = get_auc(y_true, probs, labelset)
        results['aucs'] = aucs

    return loss, results


def embed(seq, embs):
    """
    embed a sentence as an average over word embeddings
    :param word2idx:
    :return:
    """
    sent_embedding = np.array([embs[i] for i in seq])
    return np.mean(sent_embedding, 0)


def write_predictions(model, seqs, golds, lengths, sentences, tidss, labelset, fname, write_probs=False):
    """

    :param model:
    :param seqs:
    :param golds:
    :param lengths:
    :param sentences:
    :param tidss:
    :param labelset:
    :param fname:
    :param write_probs: if True, write the prediction probabilities
    :return:
    """
    y_true = golds
    probs = model.predict_proba(seqs)
    y_pred = np.argmax(probs, 1)
    outlines = [['tid', 'seq', 'gold', 'pred', 'TP']]

    for tid, sent, gold, pred in zip(tidss, sentences, y_true, y_pred):
        outlines.append(['#'+tid, sent, labelset[gold], labelset[pred], int(gold==pred)])

    if write_probs is True:
        outlines[0].extend(labelset)
        for pred, outline in zip(list(probs), outlines[1:]):
            print(pred)
            outline.extend([elm for elm in pred])

    param_reader.write_csv(fname, outlines)
    print(outlines)
    return outlines




def main(args):
    # read params from csv and update the arguments
    if args.hyperparam_csv != '':
        csv_params = param_reader.read_hyperparams_from_csv(args.hyperparam_csv, args.rowid)
        vars(args).update(csv_params)

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    seed = args.seed
    num_epochs = args.epochs
    embedding_dim = args.emb_dim

    lr = args.lr

    use_pretrained_embeddings = args.embs
    datafile = config.get('Files', 'data_{}'.format(args.data_split))
    max_vocab = args.max_vocab
    additional_data_file = args.additional_data

    torch.manual_seed(seed)
    np.random.seed(seed)

    random_name = uuid.uuid4().hex
    setup_logging(logfile=os.path.join(args.log_dir, '{}.log'.format(random_name)))
    pred_file = os.path.join(args.pred_dir, '{}.preds'.format(random_name))

    vars(args).update({'pred_file':pred_file})
    log_params(vars(args))


    feature_extractor = sents2seqs

    if use_pretrained_embeddings is True:
        embeddings_file = config.get('Files', 'emb_file')
        logging.info('Loading pretrained embedding from {}'.format(embeddings_file))
        pretrained_embeddings, word2idx, idx2word = load_embeddings_from_file(embeddings_file, max_vocab=max_vocab)
    else:
        pretrained_embeddings, word2idx, idx2word = None, None, None

    d = load_json(datafile)

    if additional_data_file != '':
        additional_data_file = config.get('Files', 'additional_data_{}'.format(args.additional_data))
        logging.info('Loading additional data from {}'.format(additional_data_file))
        additional_data = load_json(additional_data_file)
    else:
        additional_data = {'seq':[], 'label':[]}
    sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['train']['seq']] + [prefix_sequence(sent, 'ru', strip_hs=args.strip) for sent in additional_data['seq']]
    labels = d['train']['label'] + additional_data['label']

    if args.upsample is True:
        logging.info('Upsampling the train data')
        sentences, labels = upsample(sentences, labels)


    dev_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['dev']['seq']]
    dev_labels = d['dev']['label']
    dev_tids = d['dev']['tid']
    dev_raw_sentences = d['dev']['seq']

    # prepare train set
    seqs, lengths, word2idx = feature_extractor(sentences, word2idx)

    logging.info('Vocabulary has {} entries'.format(len(word2idx)))
    logging.info(word2idx)
    logging.info('Embedding sequences')
    seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in seqs])
    golds, labelset = prepare_labels(labels, None)

    # prepare dev set
    dev_seqs, dev_lengths, _ = feature_extractor(dev_sentences, word2idx)
    dev_seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in dev_seqs])
    dev_golds, _ = prepare_labels(dev_labels, labelset)
    labelnameset = ['1', '2', '3']
    print(golds.numpy())

    if args.classifier == 'logreg':
        model = LogisticRegression(solver="liblinear")

    elif args.classifier == 'svm':
        model = SVC(probability=True, kernel='poly')

    model.fit(seqs, golds.numpy())
    #labelset = model.classes_
    probs_dev = model.predict_proba(dev_seqs)
    print(probs_dev)
    dev_loss, dev_results = evaluate_validation_set(model=model, seqs=dev_seqs, golds=dev_golds, lengths=dev_lengths,
                                                sentences=dev_sentences,  labelset=labelnameset, compute_auc=True)

    logging.info('Summary dev')
    logging.info(print_result_summary(dev_results))
    logging.info(print_auc_summary(dev_results['aucs'], labelset=labelnameset))

    dev_results['best_epoch'] = 'missing'
    dev_results['best_macro_f'] = 'missing'

    param_reader.write_results_and_hyperparams(args.result_csv, dev_results, vars(args), labelnameset)
    write_predictions(model, dev_seqs, dev_golds, dev_lengths, dev_raw_sentences, dev_tids, labelnameset, pred_file, write_probs=True)

    if args.predict_test is True:
        # Prepare test data
        test_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in d['test']['seq']]
        test_labels = d['test']['label']

        test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)
        test_seqs = np.vstack([embed(seq, pretrained_embeddings) for seq in test_seqs])
        test_golds, _ = prepare_labels(test_labels, labelset)
        test_tids = d['test']['tid']
        test_raw_sentences = d['test']['seq']
        test_loss, test_results = evaluate_validation_set(model=model, seqs=test_seqs, golds=test_golds,
                                                        lengths=test_lengths,
                                                        sentences=test_sentences,
                                                        labelset=labelnameset, compute_auc=True)
        logging.info('Summary test')
        logging.info(print_result_summary(test_results))
        param_reader.write_results_and_hyperparams(args.test_result_csv, test_results, vars(args), labelnameset)
        write_predictions(model, test_seqs, test_golds, test_lengths, test_raw_sentences, test_tids, labelnameset, pred_file + '.test',
                          write_probs=True)

    if args.predict_all is True:
        # prepare the data to be predicted
        pred_data = load_json(config.get('Files', 'unlabeled'))
        test_sentences = [prefix_sequence(sent, 'en', strip_hs=args.strip) for sent in pred_data['seq']]
        test_seqs, test_lengths, _ = feature_extractor(test_sentences, word2idx)

        test_tids = pred_data['tid']
        test_raw_sentences = pred_data['seq']
        logging.info('Predicting the unlabeled data')
        write_predictions(model, test_seqs, torch.LongTensor(np.array([0 for elm in test_seqs])), test_lengths, test_raw_sentences, test_tids, labelnameset,
                          pred_file + '.unlabeled',
                          write_probs=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Tweet classification using Logistic Regression')

    parser.add_argument('--additional_data', type=str,
                        default='',
                        choices = ['', 'combi', 'sbil', 'skaschi', 'samo'],
                        help="Additional train data. if empty string, no additional data is used")
    parser.add_argument('--config', type=str, default='../config.cfg',
                        help="Config file")
    parser.add_argument('--exp_dir', type=str, default='out',
                        help="Path to experiment folder")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of epochs")
    parser.add_argument('--emb_dim', type=int, default=300,
                        help="Embedding dimension")
    parser.add_argument('--upsample', type=bool_flag, default=True,
                        help="if enabled upsample the train data according to size of largest class")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument('--embs', type=bool, default=True,
                        help="Use pre-trained embeddings")
    parser.add_argument('--max_vocab', type=int, default=-1,
                        help="Maximum number of words read in from the pretrained embeddings. -1 to disable")
    parser.add_argument('--hyperparam_csv', type=str,
                        default='../hyperparams/hyperparams_logreg.csv',
                        help="File with hyperparams. If set, values for specified hyperparams are read from the csv")
    parser.add_argument('--result_csv', type=str,
                        default='../results/results_logreg.csv',
                        help="File the results and hyperparams are written to")
    parser.add_argument('--test_result_csv', type=str,
                        default='../results/results_logreg_test.csv',
                        help="File the results and hyperparams are written to")
    parser.add_argument('--pred_dir', type=str,
                        default='../predictions/',
                        help="Directory storing the prediction files")
    parser.add_argument('--log_dir', type=str,
                        default='../logs/',
                        help="Directory storing the log files")
    parser.add_argument('--rowid', type=int,
                        default=2,
                        help="Row from which hyperparams are read")
    parser.add_argument('--data_split', type=int,
                        default=42,
                        help="Row from which hyperparams are read")
    parser.add_argument('--predict_test', type=bool_flag,
                        default=True,
                        help="Predict the test set")
    parser.add_argument('--predict_all', type=bool_flag,
                        default=False,
                        help="Predict the set of all tweets")
    parser.add_argument('--strip', type=bool_flag,
                        default=True,
                        help="Strip hashtags from words to attempt reducing oov")
    parser.add_argument('--classifier', type=str,
                        default='logreg',
                        choices = ['logreg', 'svm'],
                        help="The classifier used")
    args = parser.parse_args()
    main(args)
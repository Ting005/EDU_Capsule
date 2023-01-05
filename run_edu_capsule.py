from tqdm import tqdm
import numpy as np
import json
import pickle as pkl
import torch
import torch.nn.functional as F
from torch import optim
from config import FLAGS, compute_device
from src.dataloader import DatasetLoader, LAPTOP_ASP_DICT, REST14_ASP_DICT
from models.edu_capsule import EDUCapsule

from src.AspEvaluation import evaluateModel, evaluate_asp_only, evaluate_sem_only
from sklearn.model_selection import train_test_split
from src.utilities import log_everything


def train_step(_model, _lambda1, _optimizer, _batch_seg, _num_word_seg, _batch_sent, _num_words_sent, _sent_order, _num_edu_sent, _asp_labels, _pol_labels):
    _model.train()
    var_inputs = [_batch_seg, _num_word_seg, _batch_sent, _num_words_sent, _sent_order, _num_edu_sent, _asp_labels, _pol_labels]
    _batch_seg, _num_word_seg, _batch_sent, _num_words_sent, _sent_order, _num_edu_sent, _asp_labels, _pol_labels = [v.to(compute_device) for v in var_inputs]

    prob_asp, prob_sentiment, _, _, _ = _model(_batch_seg, _num_word_seg, _batch_sent, _num_words_sent, _sent_order, _num_edu_sent)
    loss_asp = F.binary_cross_entropy(prob_asp, _asp_labels)
    loss_sent = F.nll_loss(torch.log(prob_sentiment.permute(0, 2, 1)), _pol_labels, ignore_index=-1)
    loss = _lambda1 * loss_asp + (1 - _lambda1) * loss_sent

    _optimizer.zero_grad()
    loss.backward()
    _optimizer.step()
    return loss.data.cpu().numpy(), prob_asp.data.cpu().numpy(), prob_sentiment.data.cpu().numpy()


def evaluate(_model, _eval_data, _msg):
    _model.eval()
    with torch.no_grad():
        y_pred_asp, y_true_asp = [], []
        y_pred_sent, y_true_sent = [], []

        for (s_id, batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent, asp_labels, pol_labels) in _eval_data:
            var_inputs = [batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent]
            batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent = [v.to(compute_device) for v in var_inputs]
            prob_asp, prob_sentiment, _, _, _ = _model(batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent)
            y_pred_asp.extend((prob_asp.cpu().numpy() > 0.5).astype(int))
            y_true_asp.extend(asp_labels.numpy())
            y_pred_sent.extend(np.argmax(prob_sentiment.cpu().numpy(), axis=-1))
            y_true_sent.extend(pol_labels.numpy())

        y_pred_asp, y_true_asp = np.array(y_pred_asp), np.array(y_true_asp)
        y_pred_sent, y_true_sent = np.array(y_pred_sent), np.array(y_true_sent)
        dict_eva = evaluateModel(y_true_asp, y_pred_asp + .0, y_true_sent, y_pred_sent)
        return dict_eva


def run():
    data_path = ''
    ASP_DICT = None
    if FLAGS.data == 'lap15':
        data_path = './data/laptop15_edu_data2idx.json'
        vocab_embed_path = './data/laptop15_vocab_glove.pkl'
        ASP_DICT = LAPTOP_ASP_DICT
    elif FLAGS.data == 'rest14':
        data_path = './data/restaurant14_edu_data2idx.json'
        vocab_embed_path = './data/restaurants14_vocab_glove.pkl'
        ASP_DICT = REST14_ASP_DICT

    assert data_path != ''

    data = json.load(open(data_path, 'r'))
    if not data.__contains__('dev'):
        train_dev, test = data['train'], data['test']
        train, dev = train_test_split(train_dev, test_size=.125, random_state=FLAGS.const_random_sate)
    else:
        train, dev, test = data['train'], data['dev'], data['test']

    vocab_embed = pkl.load(open(vocab_embed_path, 'rb'))
    glove = vocab_embed['glove']
    vocab = vocab_embed['vocab']
    asp2idx = np.array([vocab[k] for k, _ in ASP_DICT.items()])
    del data
    del vocab
    del vocab_embed
    # build loading
    train_loader = DatasetLoader(FLAGS, train, is_sent=False, dataset=FLAGS.data)
    dev_loader = DatasetLoader(FLAGS, dev, is_sent=False, dataset=FLAGS.data)
    test_loader = DatasetLoader(FLAGS, test, is_sent=False, dataset=FLAGS.data)
    # get asp2idx
    model = EDUCapsule(FLAGS.dim_word, FLAGS.dim_hidden, FLAGS.n_layer, FLAGS.n_label, FLAGS.n_aspect,
                       glove, asp2idx,FLAGS.embed_dropout, FLAGS.cell_dropout, FLAGS.final_dropout,
                       FLAGS.bidirectional, compute_device)
    model.to(compute_device)

    optimizer = getattr(optim, FLAGS.optim_type)([{'params': model.base_params, 'weight_decay': FLAGS.weight_decay}], lr=FLAGS.learning_rate)

    curr_acc, best_acc, curr_avg_f1, best_avg_f1 = {'train': 0, 'test': 0, 'dev': 0}, {'train': 0, 'test': 0, 'dev': 0}, \
                                                   {'train': 0, 'test': 0, 'dev': 0}, {'train': 0, 'test': 0, 'dev': 0}
    check_point_count = 0
    with tqdm(total=100, desc='') as pbar:
        for epoch in range(0, 5000):
            for step, (s_id, batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent, asp_labels, pol_labels) in enumerate(train_loader.Data, start=1):
                losses, _, _ = train_step(model, 0.5, optimizer, batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent, asp_labels, pol_labels)

                if step % FLAGS.per_checkpoint == 0:
                    if check_point_count > 20:
                        for data_name, eval_data in zip(['train', 'dev', 'test'],
                                                        [train_loader, dev_loader, test_loader]):
                            dict_eva = evaluate(model, eval_data.Data, _msg=data_name)
                            acc = dict_eva['All']['acc']
                            avg_f1 = np.mean(dict_eva['All']['f1'][1:])

                            curr_avg_f1[data_name] = avg_f1
                            curr_acc[data_name] = acc
                            best_acc[data_name] = max(acc, best_acc[data_name])
                            best_avg_f1[data_name] = max(avg_f1, best_avg_f1[data_name])

                    pbar.update(1)
                    check_point_count += 1


if __name__ == '__main__':
    run()





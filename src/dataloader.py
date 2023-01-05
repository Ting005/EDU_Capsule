import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pdb
from collections import Counter
import re

# LAPTOP_ASP_DICT = {'laptop': 0, 'support': 1, 'battery': 2, 'company': 3, 'display': 4, 'software': 5, 'keyboard': 6,
#                    'mouse': 7, 'os': 8, 'graphics': 9, 'multimedia_devices': 10, 'hard_disc': 11, 'power_supply': 12,
#                    'shipping': 13, 'memory': 14, 'cpu': 15, 'motherboard': 16, 'warranty': 17, 'ports': 18,
#                    'hardware': 19, 'fans_cooling': 20, 'optical_drives': 21}

LAPTOP_ASP_DICT = {'laptop': 0, 'support': 1, 'display': 2, 'battery': 3, 'company': 4, 'software': 5,
                   'keyboard': 6, 'os': 7, 'mouse': 8, 'multimedia_devices': 9, 'graphics': 10, 'cpu': 11,
                   'memory': 12, 'warranty': 13}

REST14_ASP_DICT = {'food': 0, 'price': 1, 'service': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}


class ReviewDataset(Dataset):
    def __init__(self, _input_data, _n_asp, _n_pol, _dataset):
        self.dataset = _input_data
        self.n_asp = _n_asp
        self.n_pol = _n_pol
        self.asp_dict = LAPTOP_ASP_DICT if _dataset == 'lap15' else REST14_ASP_DICT

    def __len__(self):
        return len(self.dataset)

    def __pol2idx(self, _pol):
        assert _pol in ['neutral', 'positive', 'negative']
        _pol2idx = -1
        if _pol == 'neutral':
            _pol2idx = 1
        elif _pol == 'positive':
            _pol2idx = 2
        elif _pol == 'negative':
            _pol2idx = 0
        return _pol2idx

    def __getitem__(self, index):
        instance = self.dataset[index]
        s_id = instance['s_id'] if instance.__contains__('E0') else instance['id']
        segs2idx = instance['segs2idx']
        len_edus_words = [len(edu) for edu in segs2idx]
        text2idx = [w for seg in segs2idx for w in seg]
        len_sent_words = len(text2idx)
        # pdb.set_trace()
        asp_labels = np.zeros(self.n_asp, dtype='int64')
        pol_labels = np.ones(self.n_asp, dtype='int64') * (-1)
        for asppol in instance['asp_pol']:
            asp, pol = asppol.split(':') if instance.__contains__('E0') else asppol.split('#')

            if self.asp_dict.__contains__(asp):
                asp_labels[self.asp_dict[asp]] = 1
                pol_labels[self.asp_dict[asp]] = (int(pol) + 1) if instance.__contains__('E0') else self.__pol2idx(pol)

        assert asp_labels.sum() > 0
        # pdb.set_trace()

        return s_id, text2idx, len_sent_words, segs2idx, len(segs2idx), len_edus_words, asp_labels, pol_labels


class DatasetLoader(object):

    def _tuple_batch_edu(self, batch_data):
        # first sorted by number of EDUs each sentence has, in reverse order; for number of words in a sentence is unsorted.
        sorted_batch = sorted(batch_data, key=lambda b: b[4], reverse=True)
        # for batch edu, sorted by number of words in each edu
        s_id, sent2idx, num_words_sent, segs2idx, num_edus_sent, len_words_edu, asp_labels, pol_labels = zip(*sorted_batch)
        # sorted by number words in edu, with sent2idx which sorted by sentence length
        seg_stat = sorted([(len(seg), s_idx, seg_idx, seg) for s_idx, lst_segs in enumerate(segs2idx)
                           for seg_idx, seg in enumerate(lst_segs)], reverse=True)

        # padding
        max_num_seg, max_num_word = num_edus_sent[0], seg_stat[0][0]
        batch_sent = list(map(lambda x: x + [0] * (max(num_words_sent) - len(x)), sent2idx))
        batch_sent = torch.Tensor(batch_sent).long()
        batch_seg = torch.zeros(len(seg_stat), max_num_word).long()

        sent_order = torch.zeros(len(s_id), max_num_seg).long()
        num_word_seg = [s[0] for s in seg_stat]

        for seg_stat_idx, (len_seg, s_idx, seg_idx, seg) in enumerate(seg_stat):
            batch_seg[seg_stat_idx, 0: len_seg] = torch.Tensor(seg).long()
            sent_order[s_idx, seg_idx] = seg_stat_idx + 1 # 0 is for padding


        num_edu_sent = torch.Tensor(num_edus_sent).long()
        num_word_seg = torch.Tensor(num_word_seg).long()

        asp_labels = torch.Tensor(asp_labels).float()
        pol_labels = torch.Tensor(pol_labels).long()

        num_words_sent = torch.Tensor(num_words_sent).long()
        # print(num_words_sent)
        #s_id, batch_t, batch_t_sent, batch_t_sent_len, sent_order, len_sents, len_segs, asp_labels, pol_labels
        return s_id, batch_seg, num_word_seg, batch_sent, num_words_sent, sent_order, num_edu_sent, asp_labels, pol_labels

    def _tuple_batch_sent(self, batch_data):
        # sorted by sentence length
        sorted_batch = sorted(batch_data, key=lambda b: b[2], reverse=True)
        lst_id, lst_text2idx, lst_text_length, _, _, _ ,lst_asp_labels, lst_pol_labels = zip(*sorted_batch)
        # padding
        padded_sents = torch.zeros(len(lst_text2idx), lst_text_length[0]).long()
        for sent_idx, len_sen in enumerate(lst_text_length):
            padded_sents[sent_idx, 0: len_sen] = torch.Tensor(lst_text2idx[sent_idx]).long()

        lst_text_length = torch.Tensor(lst_text_length).long()
        lst_asp_labels = torch.Tensor(lst_asp_labels).float()
        lst_pol_labels = torch.Tensor(lst_pol_labels).long()
        # lst_asp2idx = torch.Tensor(lst_asp2idx).long()
        # pdb.set_trace()

        return lst_id, padded_sents, lst_text_length, lst_asp_labels, lst_pol_labels

    def filter_no_aspect_sentence(self, input_data):
        lst_output = []
        for instance in input_data:
            found = False
            for asppol in instance['asp_pol']:
                asp, pol = asppol.split('#')
                if LAPTOP_ASP_DICT.__contains__(asp):
                    found = True
                    break
            if found:
                lst_output.append(instance)
        return lst_output

    def __init__(self, FLAGS, input_data, is_sent=True, dataset='rest15'):
        self.Flag = FLAGS
        if dataset == 'lap15':
            input_data = self.filter_no_aspect_sentence(input_data)
        _dataset = ReviewDataset(input_data, FLAGS.n_aspect, FLAGS.n_label, _dataset=dataset)
        print(len(_dataset) // FLAGS.batch_size)
        self.Data = DataLoader(_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0,
                               collate_fn= self._tuple_batch_sent if is_sent else self._tuple_batch_edu, pin_memory=True)


if __name__ == '__main__':
    ''' build data2idx
    from datamanager import DataManager
    from config import FLAGS
    root = './data'
    dataset_name = ('train', 'valid', 'test')
    datamanager = DataManager(FLAGS)
    data = {}
    for tmp in dataset_name:
        data[tmp] = datamanager.load_data(FLAGS.data_dir, '%s.txt' % tmp)

    vocab, vectors = build_vocab(data['train'] + data['valid'] + data['test'])
    train2idx = data2idx(data['train'], vocab)
    dev2idx = data2idx(data['valid'], vocab)
    test2idx = data2idx(data['test'], vocab)

    processed_data = {'vocab': vocab, 'glove': vectors, 'train': train2idx, 'dev': dev2idx, 'test': test2idx}
    import pickle
    # pickle.dump(processed_data, open('./data/yq_data/yq_data2idx.pkl', 'wb'))
    print('finished')
    '''
    import json
    from config import FLAGS

    data = json.load(open('../data/restaurant15_edu_data2idx.json', 'r'))
    train, test = data['train'], data['test']
    FLAGS.batch_size = 2
    train_loader = DatasetLoader(FLAGS, train, dataset=FLAGS.data)
    test_loader = DatasetLoader(FLAGS, test, dataset=FLAGS.data)

    for data_loader in [train_loader, test_loader]:
        for step, (lst_id, padded_sents, lst_text_length, lst_asp_labels, lst_pol_labels) in enumerate(data_loader.Data, start=1):
            print(lst_asp_labels.sum())
            # pdb.set_trace()
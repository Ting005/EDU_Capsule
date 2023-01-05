# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import math
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from layers.DynamicRNN import DynamicRNN
import pdb
from torch.optim import AdamW


class CapNet(nn.Module):
    def __init__(self, n_cap, n_label, compute_device=None):
        super(CapNet, self).__init__()
        self.n_cap = n_cap
        self.compute_device = compute_device
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert/saved_model', return_dict=True, local_files_only=True)
        self.Capsules = nn.ModuleList()

        for v in range(self.n_cap):
            capsule = Capsule(asp_idx=v, n_label=n_label, n_asp=n_cap, dropout_rate=0.5, compute_device=compute_device)
            self.Capsules.append(capsule)
            # self.add_module('cap_%s' % v,
            #                 Capsule(asp_idx =v, n_label=n_label, n_asp=n_cap, dropout_rate=0., compute_device=compute_device)
            #                 )
        ignored_params = list(map(id, self.bert.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        # self.optimizer = self.configure_optimizers(learning_rate_b=2e-5, lr=2e-5, weight_decay=.0, is_freeze_bert=True)
        self.optimizer = self.configure_optimizers(learning_rate_b=2e-5, lr=1e-3, weight_decay=.0, is_freeze_bert=False)

    # _cap_aspect_ids, sent_input_ids, sent_input_ids_mask, edu_input_ids, edu_input_ids_mask, num_edus
    def forward(self, asp2idx, sent_input_ids, sent_input_ids_mask, edu_input_ids, edu_input_ids_mask, num_edus):
        # assert num_edus.sum().item() == edu_input_ids.size()[0]

        asp2idx_input_ids = asp2idx['input_ids']
        asp2idx_input_ids_mask = asp2idx['attention_mask']

        list_prob, list_prob_s, list_alpha_asp, lst_sentence_att_score, lst_edu_word_score = [], [], [], [], []
        for i in range(self.n_cap):
            asp_input_id = torch.LongTensor(asp2idx_input_ids[i]).unsqueeze(dim=0).to(self.compute_device)
            asp_input_id_mask = torch.LongTensor(asp2idx_input_ids_mask[i]).unsqueeze(dim=0).to(self.compute_device)
            # asp_input_id = torch.LongTensor(asp2idx_input_ids[i]).unsqueeze(dim=0).to(self.compute_device)
            # asp_input_id_mask = torch.LongTensor(asp2idx_input_ids_mask[i]).unsqueeze(dim=0).to(self.compute_device)
            cap_asp_prob, cap_sem_prob, kls_att_scores = self.Capsules[i](asp_input_id, asp_input_id_mask,
                                                                           sent_input_ids, sent_input_ids_mask,
                                                                           edu_input_ids, edu_input_ids_mask,
                                                                           num_edus, self.bert)
            # cap_asp_prob, cap_sem_prob, kls_att_scores = getattr(self, 'cap_%s' % i)(
            #                                                     asp_input_id, asp_input_id_mask,
            #                                                     sent_input_ids, sent_input_ids_mask,
            #                                                     edu_input_ids, edu_input_ids_mask,
            #                                                     num_edus, self.bert)

            # pdb.set_trace()
            list_prob.append(cap_asp_prob)
            list_prob_s.append(cap_sem_prob.unsqueeze(dim=1))
            list_alpha_asp.append(kls_att_scores)
        # pdb.set_trace()
        prob = torch.cat(list_prob, dim=1)
        prob_sentiment = torch.cat(list_prob_s, dim=1)
        return prob, prob_sentiment, list_alpha_asp
        # pdb.set_trace()

    def configure_optimizers(self, weight_decay, learning_rate_b, lr=1e-3, adam_epsilon=1e-8, is_freeze_bert=False):
        if not is_freeze_bert:
            "Prepare optimizer"
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                    "lr": learning_rate_b
                },
                {
                    "params": [p for n, p in self.bert.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": learning_rate_b
                },
                {
                    "params": self.base_params,
                    "weight_decay": 0.0,
                    "lr": lr
                }
            ]
        else:
            for param in self.bert.parameters():
                param.requires_grad = False
            optimizer_grouped_parameters = [{
                    "params": self.base_params,
                    "weight_decay": 0.0,
                    "lr": lr
                }]

        optimizer = AdamW(optimizer_grouped_parameters, eps=adam_epsilon)
        return optimizer


class Capsule(nn.Module):
    def __init__(self, asp_idx, n_label, n_asp, dropout_rate, compute_device):
        super(Capsule, self).__init__()
        # dim_rep = dim_hidden * (2 if flag_bid else 1)
        self.n_asp = n_asp
        self.asp_idx = asp_idx
        self.asp_vector = None
        self.caps_asp_dist = None
        dim_hidden = 768
        # dict_asp = {'food': 0, 'price': 1, 'service': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}
        # self.edu_encoder = EduEncoder(input_size=dim_hidden * 2, hidden_size=dim_hidden)
        # self.sentence_context = nn.Linear(dim_hidden * 2, dim_hidden)
        # self.edu_context = nn.Linear(dim_hidden * 2, dim_hidden)

        self.edu_asp_dist = nn.Linear(in_features=dim_hidden, out_features=self.n_asp)
        self.linear_asp = nn.Linear(dim_hidden * 2, 1)
        self.linear_sen = nn.Linear(dim_hidden * 2, n_label)
        self.edu_asp_dist = nn.Linear(in_features=dim_hidden, out_features=self.n_asp)
        # self.edu_pol_dist = nn.Linear(in_features=dim_hidden, out_features=n_label)
        self.rnn_sem = nn.LSTM(hidden_size=dim_hidden, input_size=dim_hidden, num_layers=1, bidirectional=False, bias=True)
        # self.final_dropout = nn.Dropout(dropout_rate)
        self.compute_device = compute_device
        self.init_values()

    def init_values(self):
        #{'food': 14, 'price': 129, 'service': 30, 'ambience': 245, 'anecdotes/miscellaneous': 5062}
        # dict_asp = {'food': 0, 'price': 1, 'service': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}
        asps = torch.zeros(size=(1, self.n_asp)).fill_(1e-4).to(self.compute_device)
        asps[0, self.asp_idx] = 1.
        self.caps_asp_dist = asps


    def __masked_softmax(self, mat, len_s):
        # print(len_s.type())
        len_s = len_s.type_as(mat.data)  # .long()
        # pdb.set_trace()
        idxes = torch.arange(0, int(len_s.max().item()), out=mat.data.new(int(len_s.max().item())).long()).unsqueeze(1)
        mask = (idxes.float() < len_s.unsqueeze(0)).float()
        mask = mask.t()
        mat_v, _ = mat.max(dim=-1)
        mat_norm = mat - mat_v.unsqueeze(dim=-1)
        # pdb.set_trace()
        exp = torch.exp(mat_norm) * mask
        sum_exp = exp.sum(dim=-1, keepdim=True) + 1e-4
        # test = exp / sum_exp.expand_as(exp)
        # pdb.set_trace()
        return exp / sum_exp.expand_as(exp)

    def __pad(self, tensor, length):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).to(tensor.device)])
        else:
            return tensor

    def __get_symmetric_kl_divergence_attention_scores(self, sent_asp_prob, len_sents):
        # using asymmetric for now
        target = self.caps_asp_dist.unsqueeze(dim=0).expand_as(sent_asp_prob)
        kl_loss = KLDivLoss(reduction='none')
        # sent_asp_prob_clamp = torch.clamp(sent_asp_prob, min=1e-4)
        kls = kl_loss(input=sent_asp_prob, target=target) * (-1)
        kls = kls.mean(dim=-1)
        # pdb.set_trace()
        kls_score = self.__masked_softmax(kls, len_sents)
        # pdb.set_trace()
        return kls_score

    def reset_parameters(self, dim_hidden):
        stdv = 1.0 / math.sqrt(dim_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def update_input_w_aspect(self, _asp_input_ids, _input_ids, _input_mask):
        def __pad_w_asp(__input, __max_len, __asp_id):
            __output = torch.cat([__input, __asp_id], dim=-1)
            # if __max_len + _asp_id.size(1) - __output.size(1) < 0:
            #     pdb.set_trace()
            _output = torch.cat([__output, torch.zeros(1, __max_len + __asp_id.size(1) - __output.size(1)).long().to(__output.device)], dim=1)
            _output_mask = (_output > 0).to(torch.long)
            return _output, _output_mask
        # needs to be: input ids [SEP] aspect[SEP]
        # _asp_id = _asp_input_ids[:, 1:]
        _input_temp = torch.masked_select(_input_ids, _input_mask.to(torch.bool)).unsqueeze(dim=0)
        # pdb.set_trace()
        _lens = _input_mask.sum(dim=-1)
        # pdb.set_trace()
        _start_idxes = torch.cumsum(torch.cat((_lens.data.new(1).zero_(), _lens[:-1])), 0)
        _input_temp_, _input_temp_mask_ = zip(*[__pad_w_asp(_input_temp.narrow(1, s, l), _lens.max().item(), _asp_input_ids) for s, l in
                                                zip(_start_idxes.data.tolist(), _lens.data.tolist())])
        _output_ids = torch.cat(_input_temp_, dim=0)
        _output_ids_mask = torch.cat(_input_temp_mask_, dim=0)

        return _output_ids, _output_ids_mask

    # _cap_aspect_ids, sent_input_ids, sent_input_ids_mask, edu_input_ids, edu_input_ids_mask, num_edus
    def forward(self, asp_input_ids, asp_input_ids_mask, sent_input_ids,
                sent_input_ids_mask, edu_input_ids, edu_input_ids_mask, num_edus, bert):
        # append asp ids into bert sentence
        assert num_edus.sum().item() == edu_input_ids.size()[0]
        sent_input_ids, sent_input_ids_mask = self.update_input_w_aspect(asp_input_ids, sent_input_ids, sent_input_ids_mask)
        edu_input_ids, edu_input_ids_mask = self.update_input_w_aspect(asp_input_ids, edu_input_ids, edu_input_ids_mask)

        sent_outputs = bert(sent_input_ids, sent_input_ids_mask, output_hidden_states=False)
        sent_context = sent_outputs['pooler_output']

        edu_outputs = bert(edu_input_ids, edu_input_ids_mask, output_hidden_states=False)
        edu_reps = edu_outputs['pooler_output']
        start_idxes = torch.cumsum(torch.cat((num_edus.data.new(1).zero_(), num_edus[:-1])), 0)
        # print(start_idxes)
        # print(edu_reps.size())
        # pdb.set_trace()
        sent_edu_reps = [self.__pad(edu_reps.narrow(0, s, l), num_edus.max().item())
                         for s, l in zip(start_idxes.data.tolist(), num_edus.data.tolist())]

        sent_edu_reps = torch.stack(sent_edu_reps, 0)
        # edu attention
        edu_asp_dist = self.edu_asp_dist(sent_edu_reps)
        kls_att_scores = self.__get_symmetric_kl_divergence_attention_scores(edu_asp_dist, num_edus)
        # edu
        sent_embeds_weighted = sent_edu_reps * kls_att_scores.unsqueeze(dim=-1)
        asp_rep = sent_embeds_weighted.sum(dim=1)
        # asp_rep = (sent_edu_reps * kls_att_scores.unsqueeze(dim=-1)).sum(dim=1)
        # edu -> sent rep
        asp_contextual_rep = torch.cat([sent_context, asp_rep], dim=-1)
        # asp_contextual_rep = self.final_dropout(asp_contextual_rep)
        cap_asp_prob = torch.sigmoid(self.linear_asp(asp_contextual_rep))
        # semt rep
        # pdb.set_trace()
        input_packed = pack_padded_sequence(input=sent_embeds_weighted, lengths=num_edus, batch_first=True)
        sem_rnn_outputs_packed, (h_n, h_c) = self.rnn_sem(input_packed)
        sem_rnn_outputs, output_len = pad_packed_sequence(sem_rnn_outputs_packed, batch_first=True)
        # sem_rnn_outputs, _ = self.rnn_sem(input=sent_edu_reps, lengths=num_edus, flag_ranked=True)
        caps_sem_prob = (sem_rnn_outputs * kls_att_scores.unsqueeze(dim=-1)).sum(dim=1)
        sent_features = torch.cat([sent_context, caps_sem_prob], dim=-1)

        # output_pad, output_len = pad_packed_sequence(output_packed, batch_first=self.batch_first)
        # _, (h_n, h_c) = self.rnn_sem(input=sent_embeds_weighted, lengths=num_edus, flag_ranked=True)
        # pdb.set_trace()
        # caps_sem_rep = torch.cat([h_n[-2, :], h_n[-1, :]], dim=-1)
        # caps_sem_rep = h_n[-1, :]
        # sent_features = torch.cat([sent_context, caps_sem_rep], dim=-1)
        # sent_features = self.final_dropout(sent_features)
        # caps_sem_rep = (sent_edu_reps * kls_att_scores.unsqueeze(dim=-1)).sum(dim=1)
        # sent_features = torch.cat([sent_context, caps_sem_rep], dim=-1)
        # sem_contextual_rep = self.final_dropout(sem_contextual_rep)
        cap_sem_prob = torch.softmax(self.linear_sen(sent_features), dim=-1)
        # pdb.set_trace()
        return cap_asp_prob, cap_sem_prob, kls_att_scores

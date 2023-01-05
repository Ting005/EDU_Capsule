from __future__ import unicode_literals, print_function, division
import math
import torch
import torch.nn as nn
from torch.nn import KLDivLoss
from models.DynamicRNN import DynamicRNN
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EmbedAttention(nn.Module):

    def _masked_softmax(self, mat, len_s):
        len_s = len_s.type_as(mat.data)  # .long()
        idxes = torch.arange(0, int(len_s.max().item()), out=mat.data.new(int(len_s.max().item())).long()).unsqueeze(1)
        mask = (idxes.float() < len_s.unsqueeze(0)).float()
        mask = mask.t()
        mat_v, _ = mat.max(dim=-1)
        mat = mat - mat_v.unsqueeze(dim=-1)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(1, True) + 1e-4
        return exp / sum_exp.expand_as(exp)

    def __init__(self, att_size):
        super(EmbedAttention, self).__init__()
        self.att_w = nn.Linear(att_size, 1, bias=False)

    def forward(self, mat, len_s):
        att = self.att_w(mat).squeeze(-1)
        out = self._masked_softmax(att, len_s).unsqueeze(-1)
        return out


class EduEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_asp, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=True):
        super(EduEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=input_size + 300, hidden_size=hidden_size, num_layers=num_layers,
                           bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.attention = EmbedAttention(hidden_size * 2 + 300)
        self.asp_dist = nn.Linear(in_features=hidden_size * 2, out_features=n_asp)

    def _reorder_embed(self, _embeds, _orders):
        _temp_embeds = F.pad(_embeds, (0, 0, 1, 0))  # adds a 0 to the top
        _reps = _temp_embeds[_orders.view(-1)]
        _reps = _reps.view(_orders.size(0), _orders.size(1), _temp_embeds.size(1))

        return _reps

    # takes in an aspect identity vectors
    def forward(self, aspect_vector, batch_t, len_segs, sent_order):
        aspect_vector = aspect_vector.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_t.size()[0], batch_t.size()[1],
                                                                               aspect_vector.size()[0])
        input_features = torch.cat([aspect_vector, batch_t], dim=-1)
        input_packed = pack_padded_sequence(input_features, lengths=len_segs, batch_first=self.batch_first)
        output_packed, _ = self.rnn(input_packed)
        outputs, len_ = pad_packed_sequence(output_packed, batch_first=True)
        outputs_features = torch.cat([aspect_vector, outputs], dim=-1)
        score = self.attention(outputs_features, len_segs)
        # pdb.set_trace()
        seg_rep = (outputs * score).sum(dim=1, keepdim=True).squeeze(dim=1)
        # pdb.set_trace()
        asp_logits = self.asp_dist(seg_rep)

        asp_prob = torch.log_softmax(asp_logits - asp_logits.max(dim=-1)[0].unsqueeze(dim=-1), dim=-1)
        sent_embeds = self._reorder_embed(seg_rep, sent_order)
        sent_asp_prob = self._reorder_embed(asp_prob, sent_order)
        sent_word_score = self._reorder_embed(score.squeeze(dim=-1), sent_order)
        # pdb.set_trace()
        return sent_embeds, sent_asp_prob, sent_word_score


class Capsule(nn.Module):
    def __init__(self, n_asp, asp_idx, dim_hidden, flag_bid, n_label, dropout_rate, compute_device):
        super(Capsule, self).__init__()
        dim_rep = dim_hidden * (2 if flag_bid else 1)
        self.asp_idx = asp_idx
        self.n_asp = n_asp
        self.asp_vector = None
        self.caps_asp_dist = None
        # dict_asp = {'food': 0, 'price': 1, 'service': 2, 'ambience': 3, 'anecdotes/miscellaneous': 4}
        self.edu_encoder = EduEncoder(input_size=dim_hidden * 2, hidden_size=dim_hidden, n_asp=n_asp)
        self.linear_asp = nn.Linear(dim_rep * 2, 1)
        self.linear_sen = nn.Linear(dim_rep * 2, n_label)
        self.rnn_sem = nn.LSTM(input_size=dim_hidden *2, hidden_size=dim_hidden, bidirectional=True, num_layers=1)
        self.final_dropout = nn.Dropout(dropout_rate)
        self.sent_attention = EmbedAttention(att_size=dim_hidden * 2 + 300)
        self.compute_device = compute_device
        self.reset_parameters(dim_hidden)
        self.init_values()

    def init_values(self):
        asps = torch.zeros(size=(1, self.n_asp)).fill_(1e-4).to(self.compute_device)
        asps[0, self.asp_idx] = 1.
        self.caps_asp_dist = asps

    def __masked_softmax(self, mat, len_s):
        len_s = len_s.type_as(mat.data)
        idxes = torch.arange(0, int(len_s.max().item()), out=mat.data.new(int(len_s.max().item())).long()).unsqueeze(1)
        mask = (idxes.float() < len_s.unsqueeze(0)).float()
        mask = mask.t()
        mat_v, _ = mat.max(dim=-1)
        mat_norm = mat - mat_v.unsqueeze(dim=-1)
        exp = torch.exp(mat_norm) * mask
        sum_exp = exp.sum(dim=-1, keepdim=True) + 1e-4
        return exp / sum_exp.expand_as(exp)

    def __get_symmetric_kl_divergence_attention_scores(self, sent_asp_prob, len_sents):
        target = self.caps_asp_dist.unsqueeze(dim=0).expand_as(sent_asp_prob)
        kl_loss = KLDivLoss(reduction='none')
        kls = kl_loss(input=sent_asp_prob, target=target) * (-1)
        kls = kls.mean(dim=-1)
        kls_score = self.__masked_softmax(kls, len_sents)
        return kls_score

    def reset_parameters(self, dim_hidden):
        stdv = 1.0 / math.sqrt(dim_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, asp_embeds, sent_context, edu_outputs, num_words_sent, num_word_seg, num_edu_sent, sent_order,
                rnn_sent_context_encoder):
        asp_vector = asp_embeds[self.asp_idx, :]
        sent_outputs, _ = rnn_sent_context_encoder(input=sent_context, lengths=num_words_sent, flag_ranked=False)
        sent_outputs_cat_asp = torch.cat([sent_outputs,
                                          asp_vector.unsqueeze(dim=0).unsqueeze(dim=0).expand(sent_outputs.size()[0],
                                                                                              sent_outputs.size()[1],
                                                                                              asp_vector.size()[0])],
                                         dim=-1)
        att_score = self.sent_attention(sent_outputs_cat_asp, num_words_sent)
        sent_context = (sent_outputs * att_score).sum(dim=1)
        sent_embeds, sent_asp_prob, edu_word_att_score = self.edu_encoder(asp_vector, edu_outputs, num_word_seg, sent_order)

        # edu relevance score
        kls_att_scores = self.__get_symmetric_kl_divergence_attention_scores(sent_asp_prob, num_edu_sent)
        # aspect representation
        sent_embeds_weighted = sent_embeds * kls_att_scores.unsqueeze(dim=-1)
        caps_asp_rep = sent_embeds_weighted.sum(dim=1)

        context_caps = torch.cat([sent_context, caps_asp_rep], dim=-1)
        context_caps = self.final_dropout(context_caps)
        # aspect prediction
        curr_asp_prob = torch.sigmoid(self.linear_asp(context_caps))
        # sentiment representation
        # weighted vector
        # sent_embedding
        input_packed = pack_padded_sequence(sent_embeds_weighted, lengths=num_edu_sent, batch_first=True)
        _, (h_n, h_c) = self.rnn_sem(input_packed)
        caps_sem_rep = torch.cat([h_n[-2, :], h_n[-1, :]], dim=-1)
        sent_features = torch.cat([sent_context, caps_sem_rep], dim=-1)
        # prediction
        sent_features = self.final_dropout(sent_features)
        sem_prob = torch.softmax(self.linear_sen(sent_features), dim=-1)
        return curr_asp_prob, sem_prob, kls_att_scores, att_score, edu_word_att_score


class CapNet(nn.Module):
    #n_aspects, n_labels, n_layers, dim_hidden, bidirectional, final_dropout_rate, self.compute_device
    def __init__(self, n_asp, n_label, n_layers, dim_hidden, flag_bid=True, dropout_rate=0, compute_device=None):
        super(CapNet, self).__init__()
        self.n_asp = n_asp
        self.compute_device = compute_device
        self.rnn = DynamicRNN(dim_hidden * (2 if flag_bid else 1), dim_hidden, n_layers,
                              dropout=0.5 if n_layers > 1 else 0,
                              bidirectional=flag_bid, rnn_type='LSTM', compute_device=compute_device)
        self.capsules = nn.ModuleList()

        for v in range(n_asp):
            caps = Capsule(n_asp, v, dim_hidden, flag_bid, n_label, dropout_rate, compute_device)
            self.capsules.append(caps)

    def forward(self, asp_embeds, sent_context, edu_outputs, num_words_sent, num_word_seg, num_edu_sent, sent_order):
        list_prob, list_prob_s, list_alpha_asp, lst_sentence_att_score, lst_edu_word_score = [], [], [], [], []
        for i in range(self.n_asp):
            prob_tmp, prob_semtiment_tmp, alpha_asp, sent_word_score, edu_word_score = self.capsules[i](asp_embeds,
                                                                                                        sent_context,
                                                                                                        edu_outputs,
                                                                                                        num_words_sent,
                                                                                                        num_word_seg,
                                                                                                        num_edu_sent,
                                                                                                        sent_order,
                                                                                                        self.rnn)

            list_prob.append(prob_tmp)
            list_prob_s.append(prob_semtiment_tmp.unsqueeze(dim=1))
            list_alpha_asp.append(alpha_asp)
            lst_sentence_att_score.append(sent_word_score)
            lst_edu_word_score.append(edu_word_score)
        # pdb.set_trace()
        prob = torch.cat(list_prob, dim=1)
        prob_sentiment = torch.cat(list_prob_s, dim=1)
        return prob, prob_sentiment, list_alpha_asp, lst_sentence_att_score, lst_edu_word_score
        # pdb.set_trace()


class EDUCapsule(nn.Module):
    '''
    Decoding the sentences in feedbacks
    Inout: sentences
    Output: sentence vectors, feedback vector
    '''
    def __init__(self,
                 dim_input,
                 dim_hidden,
                 n_layers,
                 n_labels,
                 n_aspects,
                 embed_list,
                 asp2idx,
                 embed_dropout_rate,
                 cell_dropout_rate,
                 final_dropout_rate,
                 bidirectional,
                 compute_device):
        super(EDUCapsule, self).__init__()

        self.asp2idx = torch.from_numpy(asp2idx)
        self.compute_device = compute_device

        self.embed = nn.Embedding.from_pretrained(embeddings=torch.Tensor(embed_list).float(), freeze=False)

        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        self.rnn_sent = DynamicRNN(input_size=dim_input, num_layers=n_layers, hidden_size=dim_hidden,
                                   dropout=(cell_dropout_rate if n_layers > 1 else 0), bidirectional=bidirectional,
                                   compute_device=compute_device)

        self.rnn_edu = nn.LSTM(input_size=dim_input, num_layers=n_layers, hidden_size=dim_hidden,
                               dropout=(cell_dropout_rate if n_layers > 1 else 0), bidirectional=bidirectional)

        self.cap_net = CapNet(n_aspects, n_labels, n_layers, dim_hidden, bidirectional,
                              final_dropout_rate, self.compute_device)

        ignored_params = list(map(id, self.embed.parameters()))
        self.base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))

    def _reorder_embed(self, _embeds, _orders):
        _embeds = F.pad(_embeds, (0, 0, 1, 0))  # adds a 0 to the top
        _reps = _embeds[_orders.view(-1)]
        _reps = _reps.view(_orders.size(0), _orders.size(1), _embeds.size(1))
        return _reps

    def forward(self, _batch_seg, _num_word_seg, _batch_sent, _num_words_sent, _sent_order, _num_edu_sent):
        seg_embeds = self.embed(_batch_seg)
        seg_embeds = self.embed_dropout(seg_embeds)
        # get sentence context vector by rnn
        sent_embeds = self.embed(_batch_sent)
        sent_embeds = self.embed_dropout(sent_embeds)
        asp_embeds = self.embed(self.asp2idx.to(self.compute_device))

        sent_embeds_pad, _ = self.rnn_sent(input=sent_embeds, lengths=_num_words_sent, flag_ranked=False)
        seg_embeds_pack = pack_padded_sequence(seg_embeds, _num_word_seg, batch_first=True)
        out_seg_pack, _ = self.rnn_edu(seg_embeds_pack)
        seg_embeds_pad, _ = pad_packed_sequence(out_seg_pack, batch_first=True)
        prob_asp, prob_sentiment, list_alpha_asp, lst_sentence_att_score, lst_edu_word_score = \
            self.cap_net(asp_embeds,
                         sent_embeds_pad,
                         seg_embeds_pad,
                         _num_words_sent,
                         _num_word_seg,
                         _num_edu_sent,
                         _sent_order)
        return prob_asp, prob_sentiment, list_alpha_asp, lst_sentence_att_score, lst_edu_word_score
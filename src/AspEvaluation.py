# coding: utf-8
# Copyright 2020, Ting Lin
# Evaluation methods, adapted from https://github.com/thuwyq/AS-Capsules
#

from __future__ import unicode_literals, print_function, division

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pdb


def evaluate_asp_only(y_true_Mat, y_pred_Mat):
    m_dict = dict()
    temp_y_true, temp_y_pred = [], []
    # evaluate aspect detection
    dict_asp = {'ovl_acc': 0, 'acc': [], 'f1': [], 'pre': [], 'rec': [], 'c_m': []}
    for i in range(np.shape(y_true_Mat)[1]):
        # pdb.set_trace()
        y_true_i = y_true_Mat[:, i]
        y_pred_i = y_pred_Mat[:, i]
        temp_y_true.extend(y_true_i)
        temp_y_pred.extend(y_pred_i)
        dict_tmp = evaluateClassification(y_true_i, y_pred_i, average='binary')
        for str_tmp in ('acc', 'f1', 'pre', 'rec', 'c_m'):
            dict_asp[str_tmp].append(dict_tmp[str_tmp])
    m_dict['Asp'] = dict_asp
    m_dict['Asp']['ovl_acc'] = round(accuracy_score(y_true=temp_y_true, y_pred=temp_y_pred), 4)
    # evaluate micro_F1 of aspect detection
    m_dict['Asp']['micro_F1_Asp'] = calculateMicroF1MultiLabel(y_true_Mat, y_pred_Mat)
    m_dict['Asp']['macro_F1_Asp'] = round(np.mean(m_dict['Asp']['f1']), 4)

    return m_dict

def evaluate_sem_only(y_true_sent, y_pred_sent):
    m_dict = dict()
    list_t_sent, list_p_sent = [], []
    for t_single, p_single in zip(y_true_sent, y_pred_sent):
        for t, p in zip(t_single, p_single):
            if t > -1:
                list_t_sent.append(t)
                list_p_sent.append(p)

    m_dict['Sen'] = evaluateClassification(list_t_sent, list_p_sent)

    return m_dict

def evaluateModel(y_true_Mat, y_pred_Mat, y_true_sent=None, y_pred_sent=None):
    m_dict = dict()

    # evaluate aspect detection
    dict_asp = {'acc': [], 'f1': [], 'pre': [], 'rec': [], 'c_m': []}
    for i in range(np.shape(y_true_Mat)[1]):
        dict_tmp = evaluateClassification(y_true_Mat[:, i], y_pred_Mat[:, i], average='binary')
        for str_tmp in ('acc', 'f1', 'pre', 'rec', 'c_m'):
            dict_asp[str_tmp].append(dict_tmp[str_tmp])
    m_dict['Asp'] = dict_asp

    # evaluate micro_F1 of aspect detection
    m_dict['micro_F1_Asp'] = calculateMicroF1MultiLabel(y_true_Mat, y_pred_Mat)

    # pdb.set_trace()
    # evaluate Sentiment with pre-annotated aspects
    list_t_sent, list_p_sent = [], []
    for t_single, p_single in zip(y_true_sent, y_pred_sent):
        for t, p in zip(t_single, p_single):
            if t > -1:
                list_t_sent.append(t)
                list_p_sent.append(p)
    m_dict['Sen'] = evaluateClassification(list_t_sent, list_p_sent)
    
    # evaluate both subtask 1 and subtask 2
    list_t_all, list_p_all = [], []
    for t_single, p_single in zip(y_true_sent, (y_pred_Mat-1) + y_pred_Mat*y_pred_sent):
        for t, p in zip(t_single, p_single):
            if (t + p) > -2:
                list_t_all.append(t)
                list_p_all.append(p)
    m_dict['All'] = evaluateClassification(list_t_all, list_p_all)

    return m_dict


def evaluateClassification(list_t, list_p, average=None):
    assert len(list_t) == len(list_p)
    c_m = confusion_matrix(list_t, list_p)
    acc = accuracy_score(list_t, list_p)
    pre = precision_score(list_t, list_p, average=average)
    rec = recall_score(list_t, list_p, average=average)
    f1 = f1_score(list_t, list_p, average=average)

    return {'c_m': c_m, 'acc': acc, 'f1': f1, 'pre': pre, 'rec': rec}


def calculateMicroF1MultiLabel(y_true_Mat, y_pred_Mat):
    tp, fp, fn = [], [], []
    for i in range(np.shape(y_true_Mat)[1]):
        list_t, list_p = y_true_Mat[:, i], y_pred_Mat[:, i]
        c_m_tmp = confusion_matrix(list_t, list_p, labels=[0, 1])
        # if len(c_m_tmp) == 1:
        #     pdb.set_trace()
        # print(c_m_tmp)
        tp.append(c_m_tmp[1][1])
        fp.append(c_m_tmp[0][1])
        fn.append(c_m_tmp[1][0])
    pre = np.sum(tp)/(np.sum(tp) + np.sum(fp))
    rec = np.sum(tp)/(np.sum(tp) + np.sum(fn))
    micro_F1 = 2*pre*rec/(pre+rec)
    # pdb.set_trace()
    return micro_F1

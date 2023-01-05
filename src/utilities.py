import logging
import numpy as np


def log_everything(_data_name, _task, _check_point, _dict_eva, _best_acc, _best_avg_f1):
    show = lambda a: '[%s]' % (' '.join(['%.4f' % x for x in a]))
    logging.info('{}@checkpoint:{}, best acc:{:.4f}, best_avg_f1:{:.4f}'.format(_data_name, _check_point, _best_acc, _best_avg_f1))
    if _task == 1:
        # pdb.set_trace()
        logging.info('Accu-Asp is %.4f, F1-Asp is %s' % (_dict_eva['Asp']['ovl_acc'], show(_dict_eva['Asp']['f1'])))
        logging.info('For Asp, Micro-F1 is {}, Macro-F1 is {}'.format(_dict_eva['Asp']['micro_F1_Asp'], _dict_eva['Asp']['macro_F1_Asp']))
        logging.info('For Asp, C_M is \n%s' % _dict_eva['Asp']['c_m'])
    elif _task == 2:
        # pdb.set_trace()
        logging.info('Accu-Sen is %.4f, F1-Sen is %s' % (_dict_eva['Sen']['acc'], show(_dict_eva['Sen']['f1'])))
        logging.info('For Sen, Micro-F1 is {}, Macro-F1 is {}'.format(_dict_eva['Sen']['f1'], np.mean(_dict_eva['Sen']['f1'][1:])))
        logging.info('For Sen, C_M is \n%s' % _dict_eva['Sen']['c_m'])
    elif _task == 3:
        logging.info('Accu-Asp is %s, F1-Asp is %s' % (_dict_eva['Asp']['acc'], show(_dict_eva['Asp']['f1'])))
        logging.info('Accu-Sen is %.4f, F1-Sen is %s' % (_dict_eva['Sen']['acc'], show(_dict_eva['Sen']['f1'])))
        logging.info('Accu-All is %.4f, F1-All is %s' % (_dict_eva['All']['acc'], show(_dict_eva['All']['f1'])))
        logging.info('For Asp, Micro-F1 is %s' % _dict_eva['micro_F1_Asp'])
        logging.info('For Sen, C_M is \n%s' % _dict_eva['Sen']['c_m'])
        logging.info('For All, C_M is \n%s' % _dict_eva['All']['c_m'])
    return None

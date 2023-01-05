import argparse, logging
import numpy as np
import random
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--dim_word', type=int, default=300, choices=[300])
parser.add_argument('--dim_hidden', type=int, default=256, choices=[256])
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--n_label', type=int, default=3, choices=[3])
parser.add_argument('--n_aspect', type=int, default=14, choices=[5, 14])
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--iter_num', type=int, default=8*320)
parser.add_argument('--per_checkpoint', type=int, default=8)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"])
parser.add_argument('--const_random_sate', type=int, default=42)
parser.add_argument('--name_model', type=str, default='edu_bert')
parser.add_argument('--model_dir', type=str, default='./runs/best_models')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--task', type=int, default=3, choices=[1, 2, 3])
parser.add_argument('--data', type=str, default='lap15', choices=['rest14', 'rest15', 'lap15'])

FLAGS = parser.parse_args()
logging.basicConfig(filename='runs/log/{}_{}_{}.log'.format(FLAGS.data, FLAGS.name_model, FLAGS.task), level=logging.INFO,
                    format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logging.info('model parameters: {}'.format(FLAGS))
torch.manual_seed(FLAGS.const_random_sate)
random.seed(FLAGS.const_random_sate)
np.random.seed(seed=FLAGS.const_random_sate)
torch.manual_seed(FLAGS.const_random_sate)
torch.cuda.manual_seed(FLAGS.const_random_sate)
torch.cuda.manual_seed_all(FLAGS.const_random_sate)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


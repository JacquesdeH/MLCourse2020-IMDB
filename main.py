import argparse
import os
import random
import numpy as np
import torch
import datetime

from Instructor import Instructor


RANDOM_SEED = 7

# data related
VALIDATION_PARTITION = 0.1
MAX_LEN = 512
NUM_WORKERS = 4

# instructor
LR = 4e-5
EPS = 1e-8
EPOCH = 3
BATCH_SIZE = 4
DISP_PERIOD = 50
MAX_NORM = 1.0
CUMUL_BATCH = 8
WARMUP_RATE = 0.1

# paths
DATA_PATH = os.path.join("data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
CKPT_PATH = os.path.join("ckpt")
LOG_PATH = os.path.join("log")

# use pretrained model
MODEL_PRETRAINED = 'roberta-base'


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=RANDOM_SEED, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--validpartition', default=VALIDATION_PARTITION, type=float)
parser.add_argument('--model_pretrained', default=MODEL_PRETRAINED, type=str)
parser.add_argument('--max_len', default=MAX_LEN, type=int)
parser.add_argument('--lr', default=LR, type=float)
parser.add_argument('--eps', default=EPS, type=float)
parser.add_argument('--epoch', default=EPOCH, type=int)
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
parser.add_argument('--num_workers', default=NUM_WORKERS, type=int)
parser.add_argument('--disp_period', default=DISP_PERIOD, type=int)
parser.add_argument('--max_norm', default=MAX_NORM, type=float)
parser.add_argument('--cumul_batch', default=CUMUL_BATCH, type=int)
parser.add_argument('--warmup_rate', default=WARMUP_RATE, type=int)

args = parser.parse_args()

args.DATA_PATH = DATA_PATH
args.RAW_PATH = RAW_PATH
args.CKPT_PATH = CKPT_PATH
args.LOG_PATH = LOG_PATH

args.DEVICE = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':
    # timestamp = "20201129-152602-RoBERTa-Cumulbatch-Alldata-Warmup"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = timestamp + '-' + "RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch"
    instructor = Instructor(model_name, args)
    # instructor.trainModel(use_all=True)
    instructor.testModel(load_pretrain=True, epoch=args.epoch)


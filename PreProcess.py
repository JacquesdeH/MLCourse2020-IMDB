# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 14:14
# Project Name: MLCourse-IMDB
# File        : PreProcess.py
# --------------------------------------------------
import os
import random

import torch
from torchtext.vocab import GloVe
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
from torch.utils.data import DataLoader, TensorDataset

SEED = 7
WORD_DIM = 100
TEST_FRAC = 0.3
UNLABEL_FRAC = 0.5
LABEL_FRAC = 0.2
SENTENCE_LEN = 500

IMDB_ROOT = './data/IMDB-movie-reviews/'


class PreProcessor:
    def __init__(self, data_path='./data/IMDB-movie-reviews/IMDB.csv'):
        self.glove = GloVe('6B', dim=WORD_DIM, cache='./.vector_cache')
        nltk.download('punkt')
        self.labeled_csv, self.unlabeled_csv, self.test_csv = self.load_divide(data_path)

    @staticmethod
    def load_divide(raw_path: str):
        raw = pd.read_csv(raw_path)
        train = raw.sample(frac=1 - TEST_FRAC, random_state=SEED)
        test = raw[~raw.index.isin(train.index)]
        labeled = train.sample(frac=LABEL_FRAC, random_state=SEED)
        unlabeled = train[~train.index.isin(labeled.index)]
        return labeled, unlabeled, test

    def digitalize(self, csv: pd.DataFrame):
        # nltk.download('punkt')
        reviews = []
        sentiments = []
        for row in csv.iterrows():
            review = row[1]['review']
            sentiment = row[1]['sentiment']
            review = nltk.word_tokenize(review)
            # review = torch.cat([glove[word.lower()].unsqueeze(dim=0) for word in review])
            # review = torch.cat([review, torch.zeros(size=[SENTENCE_LEN-review.shape[0], WORD_DIM], dtype=torch.float32)])
            # review = review.tolist()
            review = [self.glove.stoi[word.lower()] if word.lower() in self.glove.stoi else -1 for word in review] + \
                     [-1] * (SENTENCE_LEN - len(review))
            sentiment = [1 if sentiment == 'positive' else 0]
            reviews.append(np.array(review[:SENTENCE_LEN]))
            sentiments.append(np.array(sentiment))
        reviews = np.stack(reviews, axis=0)
        sentiments = np.stack(sentiments, axis=0)
        return TensorDataset(torch.LongTensor(reviews),
                             torch.LongTensor(sentiments)), reviews, sentiments

    def labeled_dataset(self):
        if os.path.exists(IMDB_ROOT+'labeled'):
            reviews = np.loadtxt(IMDB_ROOT+'labeled/reviews.csv', delimiter=',')
            sentiments = np.loadtxt(IMDB_ROOT+'labeled/sentiments.csv', delimiter=',')
            sentiments = np.expand_dims(sentiments, axis=1)
            return TensorDataset(torch.LongTensor(reviews), torch.LongTensor(sentiments))
        dataset, reviews, sentiments = self.digitalize(self.labeled_csv)
        os.makedirs(IMDB_ROOT + 'labeled')
        np.savetxt(IMDB_ROOT+'labeled/reviews.csv', reviews, delimiter=',')
        np.savetxt(IMDB_ROOT+'labeled/sentiments.csv', sentiments, delimiter=',')
        return dataset

    def unlabeled_dataset(self):
        if os.path.exists(IMDB_ROOT + 'unlabeled'):
            reviews = np.loadtxt(IMDB_ROOT + 'unlabeled/reviews.csv', delimiter=',')
            sentiments = np.loadtxt(IMDB_ROOT + 'unlabeled/sentiments.csv', delimiter=',')
            sentiments = np.expand_dims(sentiments, axis=1)
            return TensorDataset(torch.LongTensor(reviews), torch.LongTensor(sentiments))
        dataset, reviews, sentiments = self.digitalize(self.unlabeled_csv)
        os.makedirs(IMDB_ROOT + 'unlabeled')
        np.savetxt(IMDB_ROOT + 'unlabeled/reviews.csv', reviews, delimiter=',')
        np.savetxt(IMDB_ROOT + 'unlabeled/sentiments.csv', sentiments, delimiter=',')
        return dataset

    def test_dataset(self):
        if os.path.exists(IMDB_ROOT + 'test'):
            reviews = np.loadtxt(IMDB_ROOT + 'test/reviews.csv', delimiter=',')
            sentiments = np.loadtxt(IMDB_ROOT + 'test/sentiments.csv', delimiter=',')
            sentiments = np.expand_dims(sentiments, axis=1)
            return TensorDataset(torch.LongTensor(reviews), torch.LongTensor(sentiments))
        dataset, reviews, sentiments = self.digitalize(self.test_csv)
        os.makedirs(IMDB_ROOT + 'test')
        np.savetxt(IMDB_ROOT + 'test/reviews.csv', reviews, delimiter=',')
        np.savetxt(IMDB_ROOT + 'test/sentiments.csv', sentiments, delimiter=',')
        return dataset

    def embed(self, data: torch.Tensor):
        """
        :param data: [BATCH_SIZE, SENTENCE_LEN]
        :return: [BATCH_SIZE, CHANNEL=1, SENTENCE_LEN, WORD_DIM]
        """
        ret = []
        for review in data.tolist():
            embed_review = []
            for word in review:
                embed_review.append(self.glove[self.glove.itos[word]].tolist() if word != -1 else torch.zeros(WORD_DIM).tolist())
            ret.append(embed_review)
        return torch.FloatTensor(ret).unsqueeze(dim=1)

    def find_most_similar(self, cur: torch.Tensor):
        cur = cur.view(1, WORD_DIM)
        similarities = torch.cosine_similarity(cur, self.glove.vectors, dim=1)
        idx = similarities.max(dim=0)[1]
        return self.glove.itos[idx]

    def print_sentence(self, cur: torch.Tensor):
        cur = cur.view(-1, WORD_DIM)
        sentence = [self.find_most_similar(cur[i]) for i in range(SENTENCE_LEN)]
        sentence = " ".join(sentence)
        return sentence


if __name__ == '__main__':
    preprocessor = PreProcessor()
    pass

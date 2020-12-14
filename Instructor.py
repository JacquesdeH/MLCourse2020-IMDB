# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 13:30
# Project Name: MLCourse-IMDB
# File        : Instructor.py
# --------------------------------------------------
import math
import os
import pandas as pd
import re
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

from IMDBDataSet import IMDBDataset

class Instructor:
    def __init__(self, model_name: str, args):
        self.model_name = model_name
        self.args = args
        self.trainDataset = None
        self.validDataset = None
        self.testDataset = None
        self.trainDataloader = None
        self.validDataloader = None
        self.testDataloader = None
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.args.model_pretrained,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        ).to(self.args.DEVICE)
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.batch_cnt = 0
        self.valid_cnt = 0

    @staticmethod
    def _rm_tags(text):
        re_tags = re.compile(r'<[^>]+>')
        return re_tags.sub(' ', text)

    def loadRawTrain(self):
        rawTrainPath = os.path.join(self.args.RAW_PATH, "train.csv")
        df = pd.read_csv(rawTrainPath)
        del df['Unnamed: 0']
        df['review'] = df['review'].apply(lambda x: self._rm_tags(x))
        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        all_data, all_label = df['review'].tolist(), df['sentiment'].tolist()
        return all_data, all_label

    def prepareTrainValid(self, use_all: bool):
        all_data, all_label = self.loadRawTrain()
        if not use_all:
            train_data, valid_data, train_label, valid_label = \
                train_test_split(all_data, all_label, random_state=self.args.seed, test_size=self.args.validpartition)
            self.trainDataset = IMDBDataset(reviews=train_data, sentiments=train_label, args=self.args)
            self.validDataset = IMDBDataset(reviews=valid_data, sentiments=valid_label, args=self.args)
            self.trainDataloader = DataLoader(dataset=self.trainDataset, batch_size=self.args.batch_size,
                                              shuffle=True, num_workers=self.args.num_workers)
            self.validDataloader = DataLoader(dataset=self.validDataset, batch_size=self.args.batch_size,
                                              shuffle=False, num_workers=self.args.num_workers)
        else:
            self.trainDataset = IMDBDataset(reviews=all_data, sentiments=all_label, args=self.args)
            self.trainDataloader = DataLoader(dataset=self.trainDataset, batch_size=self.args.batch_size,
                                              shuffle=True, num_workers=self.args.num_workers)

    def loadRawTest(self):
        rawTestPath = os.path.join(self.args.RAW_PATH, "test_data.csv")
        df = pd.read_csv(rawTestPath)
        df['review'] = df['review'].apply(lambda x: self._rm_tags(x))
        df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        del df['sentiment']
        test_ids, test_data = df['Unnamed: 0'], df['review']
        return test_ids, test_data

    def prepareTest(self):
        test_ids, test_data = self.loadRawTest()
        self.testDataset = IMDBDataset(reviews=test_data, original_ids=test_ids, args=self.args)
        self.testDataloader = DataLoader(dataset=self.testDataset, batch_size=self.args.batch_size,
                                         shuffle=False, num_workers=self.args.num_workers)

    def trainModel(self, use_all=False):
        self.writer = SummaryWriter(os.path.join(self.args.LOG_PATH, self.model_name))
        self.prepareTrainValid(use_all)
        epochs = self.args.epoch
        tot_steps = math.ceil(len(self.trainDataloader) / self.args.cumul_batch) * epochs
        warmup_steps = tot_steps * self.args.warmup_rate
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=self.args.eps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=tot_steps)
        for epoch in range(1, epochs+1):
            print()
            print("================ Training Epoch {:}/{:} ================".format(epoch, epochs))
            print("Start training ------>")
            self.epochTrain(epoch)
            print()
            if not use_all:
                print("---------------- Validing Epoch {:}/{:} ----------------".format(epoch, epochs))
                self.epochValid()
                print()
        self.writer.close()

    def epochTrain(self, epoch):
        self.model.train()

        cumul_loss = 0
        cumul_acc = 0
        cumul_steps = 0
        cumul_samples = 0

        self.optimizer.zero_grad()
        cumulative_batch = 0

        for idx, (input_ids, mask_attentions, labels) in enumerate(tqdm(self.trainDataloader)):
            batch_size = labels.shape[0]

            input_ids, mask_attentions, labels = input_ids.to(self.args.DEVICE), mask_attentions.to(
                self.args.DEVICE), labels.to(self.args.DEVICE)
            loss, outputs = self.model(input_ids, attention_mask=mask_attentions, labels=labels)

            loss_each = loss / self.args.cumul_batch
            loss_each.backward()

            pred = outputs.max(dim=1)[1].unsqueeze(dim=1)

            cumulative_batch += 1
            cumul_steps += 1
            cumul_loss += loss.detach().cpu().item() * batch_size
            cumul_acc += (pred == labels).type(torch.float).sum().cpu().item()
            cumul_samples += batch_size

            if cumulative_batch >= self.args.cumul_batch:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                cumulative_batch = 0

            if cumul_steps >= self.args.disp_period or idx+1 == len(self.trainDataloader):
                print(" -> cumul_steps={:} loss={:} acc={:}".format(cumul_steps, cumul_loss/cumul_samples, cumul_acc/cumul_samples))
                self.batch_cnt += 1
                self.writer.add_scalar('batch-loss', cumul_loss/cumul_samples, global_step=self.batch_cnt)
                self.writer.add_scalar('batch-acc', cumul_acc/cumul_samples, global_step=self.batch_cnt)
                self.writer.add_scalar('learning-rate', self.optimizer.state_dict()['param_groups'][0]['lr'], global_step=self.batch_cnt)
                cumul_steps = 0
                cumul_loss = 0
                cumul_acc = 0
                cumul_samples = 0

        self.save(epoch)

    def epochValid(self):
        self.model.eval()

        cumul_samples = 0
        cumul_loss = 0
        cumul_acc = 0

        for idx, (input_ids, mask_attentions, labels) in enumerate(tqdm(self.validDataloader)):
            batch_size = labels.shape[0]
            input_ids, mask_attentions, labels = input_ids.to(self.args.DEVICE), mask_attentions.to(
                self.args.DEVICE), labels.to(self.args.DEVICE)
            with torch.no_grad():
                loss, outputs = self.model(input_ids, attention_mask=mask_attentions, labels=labels)
                pred = outputs.max(dim=1)[1].unsqueeze(dim=1)
                cumul_loss += loss.cpu().item() * batch_size
                cumul_acc += (pred == labels).type(torch.float).sum().cpu().item()
                cumul_samples += batch_size

        print(" ----> Eval_loss={:} Eval_acc={:}".format(cumul_loss/cumul_samples, cumul_acc/cumul_samples))
        self.valid_cnt += 1
        self.writer.add_scalar('valid_loss', cumul_loss/cumul_samples, global_step=self.valid_cnt)
        self.writer.add_scalar('valid-acc', cumul_acc/cumul_samples, global_step=self.valid_cnt)

    def testModel(self, load_pretrain=False, epoch=0):
        if load_pretrain:
            self.load(epoch)
        self.prepareTest()
        id2ans = {}
        for idx, (input_ids, mask_attentions, original_ids) in enumerate(tqdm(self.testDataloader)):
            input_ids, mask_attentions, original_ids = input_ids.to(self.args.DEVICE), mask_attentions.to(
                self.args.DEVICE), original_ids.to(self.args.DEVICE)
            with torch.no_grad():
                outputs, = self.model(input_ids, attention_mask=mask_attentions)
                pred = outputs.max(dim=1)[1].unsqueeze(dim=1)
                for id_, ans in zip(original_ids, pred):
                    id2ans[id_.item()] = ans.item()
        submission = self.generate_submission(id2ans)
        submission_file = os.path.join(self.args.DATA_PATH, "submission.csv")
        submission.to_csv(submission_file, index=False, index_label=False)
        print('  -> Done submission generation of {:}'.format(self.model_name))

    def generate_submission(self, id2ans: dict):
        rawTestPath = os.path.join(self.args.RAW_PATH, "test_data.csv")
        df = pd.read_csv(rawTestPath)
        del df['review']
        df['sentiment'] = df['Unnamed: 0'].apply(lambda entry: 'positive' if id2ans[entry] == 1 else 'negative')
        return df

    def save(self, epoch):
        filepath = os.path.join(self.args.CKPT_PATH, self.model_name + "--EPOCH-{:}".format(epoch))
        print("-----------------------------------------------")
        print("  -> Saving model {:} ......".format(filepath))
        torch.save(self.model.state_dict(), filepath)
        print("  -> Successfully saved model.")
        print("-----------------------------------------------")

    def load(self, epoch):
        filepath = os.path.join(self.args.CKPT_PATH, self.model_name + "--EPOCH-{:}".format(epoch))
        if not os.path.exists(filepath):
            print("  -> Unable to load from filepath {:} !".format(filepath))
            return
        self.model.load_state_dict(torch.load(filepath))
        print("  -> Loaded from model {:}".format(filepath))


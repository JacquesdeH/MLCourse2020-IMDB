# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/11/27 14:14
# Project Name: MLCourse-IMDB
# File        : IMDBDataSet.py
# --------------------------------------------------

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from keras.preprocessing.sequence import pad_sequences


class IMDBDataset(Dataset):
    def __init__(self, reviews, sentiments=None, original_ids=None, args=None):
        super(IMDBDataset, self).__init__()
        self.args = args
        self.reviews = reviews
        self.sentiments = sentiments
        self.original_ids = original_ids
        self.length = len(reviews)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_pretrained, do_lower_case=True)

    def __getitem__(self, index):
        review = self.reviews[index]
        input_ids = self.tokenizer.encode(review, add_special_tokens=True, max_length=self.args.max_len, truncation=True)
        input_ids = pad_sequences([input_ids], maxlen=self.args.max_len, dtype="long",
                                  value=0, truncating="post", padding="post")
        input_ids = torch.tensor(input_ids.squeeze(axis=0), dtype=torch.long)
        attention_masks = torch.tensor([int(token_id > 0) for token_id in input_ids], dtype=torch.long)
        if self.sentiments is None:
            original_id = self.original_ids[index]
            original_id = torch.tensor([original_id], dtype=torch.long)
            return input_ids, attention_masks, original_id
        else:
            sentiment = self.sentiments[index]
            sentiment = torch.tensor([sentiment], dtype=torch.long)
            return input_ids, attention_masks, sentiment

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import main
    dataset = IMDBDataset(["Good for you!", "What the hell?"], [1, 0], main.args)
    for x, mask, y in dataset:
        pass

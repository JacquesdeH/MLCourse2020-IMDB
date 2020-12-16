# _*_coding:utf-8_*_
# Author      : JacquesdeH
# Create Time : 2020/12/15 10:56
# Project Name: MLCourse-IMDB
# File        : Ensembler.py
# --------------------------------------------------

import os
import pandas as pd

RAW_SUBMISSION_PATH = os.path.join(os.path.join("data", "raw"), "submission.csv")
ENSEMBLE_PATH = os.path.join("data", "ensemble")
DST_SUBMISSION = os.path.join(ENSEMBLE_PATH, "submission.csv")

submissions = {
    "submission-20201205-194227-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.50,
    "submission-20201214-215325-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.63,
    "submission-20201215-082241-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.50,
    "submission-20201215-110700-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.41,
    "submission-20201216-000133-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.24,
    "submission-20201216-142158-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.45,
    "submission-20201216-165851-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.51,
}

emotions = ["positive", "negative"]
raw = pd.read_csv(RAW_SUBMISSION_PATH)
files = raw['Unnamed: 0'].tolist()
ensemble_dict = dict([(filename, dict([(emotion, 0.0) for emotion in emotions])) for filename in files])

for submission, score in submissions.items():
    csv_path = os.path.join(ENSEMBLE_PATH, os.path.join(submission, "submission.csv"))
    df = pd.read_csv(csv_path)
    for filename, emotion in zip(df['Unnamed: 0'], df['sentiment']):
        ensemble_dict[filename][emotion] += score

ensembled_pairs = dict([(filename, max(emotion_vec.items(), key=lambda x: x[1])[0])
                        for (filename, emotion_vec) in ensemble_dict.items()])

ensembled_df = pd.DataFrame(data={"Unnamed: 0": list(ensembled_pairs.keys()),
                                  "sentiment": list(ensembled_pairs.values())})
ensembled_df.to_csv(DST_SUBMISSION, index=False, index_label=False)

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
    # "submission-20201205-194227-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.50,
    # "submission-20201214-215325-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.63,
    # "submission-20201215-082241-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.50,
    # "submission-20201215-110700-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.41,
    # "submission-20201216-000133-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.24,
    # "submission-20201216-142158-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.45,
    # "submission-20201216-165851-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.51,
    # "submission-20201216-220908-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.37,
    # "submission-20201217-011328-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.61,
    # "submission-20201225-171648-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.41,
    # "submission-20201226-011026-RoBERTa-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.46,
    "submission-20201226-115507-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.13,
    "submission-20201226-182046-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.08,
    # "submission-20201227-134848-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.90,
    # "submission-20201227-202207-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.94,
    "submission-20201227-235217-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.01,
    # "submission-20201228-000216-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.90,
    "submission-20201228-134621-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.15,
    # "submission-20201228-154949-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.99,
    "submission-20201228-182820-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.02,
    # "submission-20201228-195811-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.86,
    "submission-20201228-200251-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.20,
    "submission-20201228-220723-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.09,
    # "submission-20201229-010306-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.99,
    "submission-20201229-020822-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 96.09,
    # "submission-20201229-023100-RoBERTa-large-Cumulbatch-Alldata-Warmup-Batchlarger-MoreEpoch": 95.90,
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

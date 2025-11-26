import csv

import pandas as pd
from simpletransformers.classification import ClassificationModel
import torch


DATA_ROOT = '../data'
LARGE_CORPUS = f'{DATA_ROOT}/large_corpus.csv'
LARGE_CORPUS_CATEGORISED = f'{DATA_ROOT}/large_corpus_scored.csv'

LABEL_TO_RATING_CLASSES = {0: 'negative', 1: 'neutral', 2: 'positive'}

MODEL_ARCHITECTURE = 'roberta'
MODEL_LOCATION = '../part-4/outputs/'


def load_large_corpus() -> pd.DataFrame:
    df = pd.read_csv(LARGE_CORPUS, quoting=csv.QUOTE_ALL, keep_default_na=False)
    df = df[["reviews"]]
    return df


def score(df: pd.DataFrame, use_cuda: bool) -> pd.DataFrame:
    model = ClassificationModel(MODEL_ARCHITECTURE, MODEL_LOCATION, num_labels=3, use_cuda=use_cuda)
    predictions, _ = model.predict(df['reviews'].to_list())
    df['score'] = list(map(lambda x: LABEL_TO_RATING_CLASSES[x], predictions))
    return df


def save_corpus(df: pd.DataFrame):
    df.to_csv(LARGE_CORPUS_CATEGORISED, index=False, quoting=csv.QUOTE_ALL)


def main():
    use_cuda = torch.cuda.is_available()

    large_corpus_df = load_large_corpus()
    large_corpus_df = score(large_corpus_df, use_cuda)
    save_corpus(large_corpus_df)


if __name__ == "__main__":
    main()
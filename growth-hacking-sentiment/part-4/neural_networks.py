import functools
import platform
import time

import matplotlib.pyplot as plt
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.language_modeling import LanguageModelingModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import pipeline

data_root = '../data'
small_corpus = f'{data_root}/small_corpus.csv'
train_txt = f'{data_root}/train.txt'
test_txt = f'{data_root}/test.txt'

labels = ['negative', 'neutral', 'positive']

RANDOM_SEED = 42


def add_ratings_class(df):
    df['ratings_class'] = df['ratings'].apply(
        lambda x: 'positive' if x >=5 else ('negative' if x <= 1 else 'neutral')  )


def model_to_classification(model_output, negative_threshold, positive_threshold):
    label = model_output[0]['label']
    score = model_output[0]['score']
    if label == 'NEGATIVE' and score > negative_threshold:
        return 'negative'
    elif label == 'POSITIVE' and score > positive_threshold:
        return 'positive'
    else:
        return 'neutral'


def accuracy_precision_recall(y_true, y_pred):
    # accuracy - number of correctly classified reviews

    # precision - for a given score class, percentage of correctly classified reviews
    # Important if the costs of a false positive (e.g spam) are high

    # recall - for given ratings class, percentage of correctly classified reviews
    # Important if the costs of a false negative (e.g fraud, disease) are high

    # f1 score - harmonic mean of precision and recall
    print(f'Accuracy score: {accuracy_score(y_true, y_pred):.2f}')
    print(f'Precision score: {precision_score(y_true, y_pred, average="weighted"):.2f}')
    print(f'Recall score: {recall_score(y_true, y_pred, average="weighted"):.2f}')
    print()


def calculate_confusion_matrix(y_true, y_pred):
    # Ground truth in rows, predictions in columns
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot()
    plt.show()


def evaluate_model(df):
    y_true = df['ratings_class']
    y_pred = df['score']

    accuracy_precision_recall(y_true, y_pred)
    calculate_confusion_matrix(y_true, y_pred)


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} secs")
        return value

    return wrapper_timer


@timer
def eval_model1(df, device):
    df = df.copy(deep=True)
    model = pipeline(model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
                     device=device)
    print("Processing reviews...")
    df['score'] = df.apply(lambda row: model_to_classification(
        model(row['reviews'][:512]), 0.998, 0.94
    ), axis=1)

    evaluate_model(df)

# With the threshold at 0.75, it is putting too many neutral reviews in the negative category
# and positive category. Should be higher.
#
# Time taken: 145 (with CPU)
# Time taken: 68 (with MPS)


# 0.998, 0.94
# Accuracy score: 0.64
# Precision score: 0.62
# Recall score: 0.64
#
# Confusion matrix
# [[1018  415   67]
#  [ 395  570  535]
#  [  32  198 1270]]
# Finished eval_model1() in 807.2092 secs
# Finished processing reviews.



RATINGS_CLASS_TO_LABEL = {'negative': 0, 'neutral': 1, 'positive': 2}

LABEL_TO_RATING_CLASSES = {0: 'negative', 1: 'neutral', 2: 'positive'}


@timer
def create_model2(train_df):
    train_df = train_df.copy(deep=True)
    train_df = train_df.rename(columns={'reviews': 'text'})
    train_df['labels'] = train_df['ratings_class'].apply(lambda x: RATINGS_CLASS_TO_LABEL[x])

    # 'distilbert', distilbert-base-uncased
    # "roberta", "roberta-base"
    # 'max_seq_length': 512,
    # 'sliding_window': True,n
    model = ClassificationModel("roberta", "roberta-base",
                                num_labels=3, use_cuda=False,
                                args={'num_train_epochs': 1, 'best_model_dir': 'models/', 'max_seq_length': 512,
                                      'overwrite_output_dir': True, 'sliding_window': True,
                                      'evaluate_during_training': False, 'train_batch_size': 20, 'eval_batch_size': 20})

    model.train_model(train_df, output_dir='models/')


@timer
def eval_model_2(test_df):
    test_df = test_df.copy(deep=True)
    # roberta
    model = ClassificationModel('roberta', 'outputs/', num_labels=3, use_cuda=False)

    print("Processing reviews...")
    predictions, _ = model.predict(test_df['reviews'].to_list())
    test_df['score'] = list(map(lambda x: LABEL_TO_RATING_CLASSES[x], predictions))

    evaluate_model(test_df)


def create_file(reviews, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(reviews))

# output_dir 	str 	“outputs/” 	The directory where all outputs will be stored. This includes model checkpoints and evaluation results.
# best_model_dir 	str 	outputs/best_model 	The directory where the best model (model checkpoints) will be saved (based on eval_during_training)


@timer
def create_model3(train_df, test_df):
    train_df = train_df.copy(deep=True)

    # Fine tune the model
    model = LanguageModelingModel('roberta', 'roberta-base', use_cuda=False,
                                  args={'num_train_epochs': 1,
                                        'overwrite_output_dir': True, 'sliding_window': True,
                                        'output_dir': 'outputs/', 'best_model_dir': 'best_model',
                                        'max_seq_length': 512})
    model.train_model(train_txt)

    # Train the classifier
    model = ClassificationModel("roberta", 'outputs/',
                                num_labels=3, use_cuda=False,
                                args={'num_train_epochs': 1, 'best_model_dir': 'models/', 'max_seq_length': 512,
                                      'overwrite_output_dir': True, 'sliding_window': True,
                                      'evaluate_during_training': False, 'train_batch_size': 20, 'eval_batch_size': 20})

    train_df = train_df.rename(columns={'reviews': 'text'})
    train_df['labels'] = train_df['ratings_class'].apply(lambda x: RATINGS_CLASS_TO_LABEL[x])

    model.train_model(train_df)


def eval_model_3(test_df):
    test_df = test_df.copy(deep=True)
    # roberta
    model = ClassificationModel('roberta', 'outputs/', num_labels=3, use_cuda=False)

    print("Processing reviews...")
    predictions, _ = model.predict(test_df['reviews'].to_list())
    test_df['score'] = list(map(lambda x: LABEL_TO_RATING_CLASSES[x], predictions))

    evaluate_model(test_df)


#
# distilbert-base-uncased
# Accuracy score: 0.69
# Precision score: 0.68
# Recall score: 0.69
#
# Confusion matrix
# [[604 114  32]
#  [225 381 144]
#  [ 37 148 565]]
# Finished eval_model_2() in 75.4800 secs


# distilbert-base-uncased 512k sequences
# Accuracy score: 0.73
# Precision score: 0.72
# Recall score: 0.73
#
# Confusion matrix
# [[606 117  27]
#  [191 409 150]
#  [ 16 114 620]]
# Finished eval_model_2() in 395.2511 secs

# roberta-base
# Accuracy score: 0.73
# Precision score: 0.73
# Recall score: 0.73
#
# Confusion matrix
# [[637  97  16]
#  [199 435 116]
#  [ 12 162 576]]
# Finished eval_model_2() in 117.7929 secs
# Finished processing reviews.


# Roberta-base 512k sequences
# Finished create_model2() in 3592.0646 secs
# Accuracy score: 0.77
# Precision score: 0.76
# Recall score: 0.77
#
# Confusion matrix
# [[646  86  18]
#  [183 428 139]
#  [  5  95 650]]
# Finished eval_model_2() in 1885.6078 secs
# Finished processing reviews.

# deberta-base
# Accuracy score: 0.72
# Precision score: 0.71
# Recall score: 0.72
#
# Confusion matrix
# [[576 144  30]
#  [163 423 164]
#  [ 12 120 618]]

# deberta-base 512k sequences
# Accuracy score: 0.75
# Precision score: 0.75
# Recall score: 0.75
#
# Confusion matrix
# [[580 118  52]
#  [141 449 160]
#  [  5  79 666]]
# Finished eval_model_2() in 1791.2783 secs
# Finished processing reviews.
# But very slow

# Roberta-base-512k sequences
# Accuracy score: 0.75
# Precision score: 0.75
# Recall score: 0.75
#
# Confusion matrix
# [[658  71  21]
#  [210 375 165]
#  [ 10  77 663]]



def main():
    print(platform.processor())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(small_corpus)
    print(f"Number of elements: {len(df)}")
    add_ratings_class(df)

    train_df, test_df = train_test_split(df, test_size=0.5, random_state=RANDOM_SEED,
                                         stratify=df['ratings_class'])
    # Load large corpus and prep it
    print(f'Size of the test set: {len(train_df)}')

    # create_model3(train_df, test_df)
    # eval_model_3(test_df)

    # eval_model1(df, device)

    # create_model2(train_df)
    # eval_model_2(test_df)

    print("Finished processing reviews.")


if __name__ == "__main__":
    main()

# Fine-tuned Roberta model
# and trained classifier


# Accuracy score: 0.75
# Precision score: 0.75
# Recall score: 0.75
#
# Confusion matrix
# [[658  71  21]
#  [210 375 165]
#  [ 10  77 663]]
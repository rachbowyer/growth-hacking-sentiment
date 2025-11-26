import csv
import functools
import platform
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.language_modeling import LanguageModelingModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from transformers import pipeline

# Suppress specific transformers warnings
warnings.filterwarnings('ignore', message='.*were not initialized from the model checkpoint.*')
warnings.filterwarnings('ignore', message='.*TRAIN this model on a down-stream task.*')

data_root = '../data'
small_corpus = f'{data_root}/small_corpus.csv'
LARGE_CORPUS = f'{data_root}/large_corpus.csv'
TRAIN_TXT = f'{data_root}/train.txt'
# TEST_TXT = f'{data_root}/test.txt'
EVAL_TXT = f'{data_root}/eval.txt'

labels = ['negative', 'neutral', 'positive']


BATCH_SIZE = 128
MAX_SEQ_LEN = 512


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
    print(f'F1 score: {f1_score(y_true, y_pred, average="weighted"):.2f}')
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
                     device=device, batch_size=BATCH_SIZE)  # Add batch_size parameter
    print("Processing reviews...")
    reviews_truncated = [review[:MAX_SEQ_LEN] for review in df['reviews']]
    results = model(reviews_truncated)

    # Apply classification thresholds
    df['score'] = [model_to_classification([result], 0.998, 0.94) for result in results]

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
def create_model2(train_df, use_cuda):
    train_df = train_df.copy(deep=True)
    train_df = train_df.rename(columns={'reviews': 'text'})
    train_df['labels'] = train_df['ratings_class'].apply(lambda x: RATINGS_CLASS_TO_LABEL[x])

    # 'distilbert', distilbert-base-uncased
    # "roberta", "roberta-base"
    # "roberta", "roberta-large"
    # "debertav2", "microsoft/deberta-v3-large"
    # 'max_seq_length': 512,
    # 'sliding_window': True,n
    model = ClassificationModel("roberta", "roberta-large",
                                num_labels=3, use_cuda=use_cuda,
                                args={'num_train_epochs': 1,
                                      'best_model_dir': 'models/',
                                      'max_seq_length': MAX_SEQ_LEN,
                                      'overwrite_output_dir': True,
                                      'sliding_window': True,  # Process long reviews in 512-token windows
                                      'train_batch_size': 20,  # Balance between speed and memory usage
                                })  # Disable mixed precision to avoid deprecated warnings

    model.train_model(train_df, output_dir='models/')


@timer
def eval_model_2(test_df, use_cuda):
    test_df = test_df.copy(deep=True)
    # roberta
    model = ClassificationModel('roberta', 'outputs/', num_labels=3, use_cuda=use_cuda)

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
def create_model3(train_df, use_cuda):
    train_df = train_df.copy(deep=True)

    # Fine tune the model
    model = LanguageModelingModel('roberta', 'roberta-base', use_cuda=use_cuda,
                                  args={'num_train_epochs': 1,
                                        'overwrite_output_dir': True, 'sliding_window': True,
                                        'output_dir': 'outputs/', 'best_model_dir': 'best_model',
                                        'max_seq_length': 512})
    model.train_model(TRAIN_TXT)

    # result, _, _ = model.eval_model(EVAL_TXT)
    # print(f"Eval loss: {result['eval_loss']:.4f}, Perplexity: {result['perplexity']:.4f}")


    # Train the classifier
    model = ClassificationModel("roberta", 'outputs/',
                                num_labels=3, use_cuda=use_cuda,
                                args={'num_train_epochs': 1, 'best_model_dir': 'models/', 'max_seq_length': 512,
                                      'overwrite_output_dir': True, 'sliding_window': True,
                                      'evaluate_during_training': False, 'train_batch_size': 20, 'eval_batch_size': 20})

    train_df = train_df.rename(columns={'reviews': 'text'})
    train_df['labels'] = train_df['ratings_class'].apply(lambda x: RATINGS_CLASS_TO_LABEL[x])

    model.train_model(train_df)


def eval_model_3(test_df, use_cuda):
    test_df = test_df.copy(deep=True)
    # roberta
    model = ClassificationModel('roberta', 'outputs/', num_labels=3, use_cuda=use_cuda)

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


# Roberta-large-512 sequences
# Accuracy score: 0.77
# Precision score: 0.77
# Recall score: 0.77
# F1 score: 0.76

# Confusion matrix
# [[585 128  37]
#  [122 465 163]
#  [  7  63 680]]
# Train time (with GPU) - 74 secs (2500 reviews)
# Eval time (with GPU) - 23 secs (2500 reviews)


# big-bird-large-4096 sequences
# Accuracy score: 0.72
# Precision score: 0.72
# Recall score: 0.72
# F1 score: 0.72

# Confusion matrix
# [[603 114  33]
#  [181 424 145]
#  [ 24 124 602]]


def create_reviews_as_txt_file(df, filename):
    df = list(df["reviews"])
    reviews = [r.strip() for r in df]
    create_file(reviews, filename)


def main():
    print(platform.processor())
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    # device = "cpu"
    print(f"Using device: {device}")

    small_corpus_df = pd.read_csv(small_corpus, quoting=csv.QUOTE_ALL, keep_default_na=False)
    print(f"Number of elements: {len(small_corpus_df)}")
    add_ratings_class(small_corpus_df)

    train_df, test_df = train_test_split(small_corpus_df, test_size=0.5, random_state=RANDOM_SEED,
                                         stratify=small_corpus_df['ratings_class'])

    review_lengths = train_df['reviews'].str.len()
    print(f"Review length - avg: {review_lengths.mean():.0f}, min: {review_lengths.min()}, max: {review_lengths.max()}")

    large_corpus_df = pd.read_csv(LARGE_CORPUS, quoting=csv.QUOTE_ALL, keep_default_na=False)
    eval_df = large_corpus_df.sample(n=100, random_state=RANDOM_SEED)

    create_reviews_as_txt_file(train_df, TRAIN_TXT)
    # create_reviews_as_txt_file(test_df, TEST_TXT)
    create_reviews_as_txt_file(eval_df, EVAL_TXT)


    print(f'Size of the train set: {len(train_df)}')
    print(f'Size of the test set: {len(test_df)}')
    print(f'Size of the eval set: {len(eval_df)}')
  
    # eval_model1(test_df, device)

    # create_model2(train_df, use_cuda)
    eval_model_2(test_df, use_cuda)

    # create_model3(train_df, use_cuda)
    # eval_model_3(test_df, use_cuda)


    print("Finished processing reviews.")


if __name__ == "__main__":
    main()


# eval_model1
# Evaluates an out of the box Hugging Faces Distilbert based classifer

# create_model2
# trains a classification head ontop of the stock RoBERTa model

# eval_model_2
# Evaluates the second model

# create_model_3
# Firstly, fine tunes RoBERTa based on the large corpus of text
# Then trains a classification head ontop of this fine tuned model

# eval_model_3
# Evaluates the third model

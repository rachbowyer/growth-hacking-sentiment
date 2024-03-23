import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


data_root = '../data'
small_corpus_scored = f'{data_root}/small_corpus_scored.csv'

labels = ['negative', 'neutral', 'positive']


def add_ratings_class(df):
    df['ratings_class'] = df['ratings'].apply(
        lambda x: 'positive' if x >=5 else ('negative' if x <= 1 else 'neutral')
    )


def add_score_class(df):
    df['score_class'] = df['score'].apply(
        lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
    )


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


def main():
    df = pd.read_csv(small_corpus_scored)
    add_ratings_class(df)
    add_score_class(df)

    y_true = df['ratings_class']
    y_pred = df['score_class']

    accuracy_precision_recall(y_true, y_pred)

    # Textual report
    print(classification_report(y_true, y_pred, labels=labels))

    # Confusion matrix
    calculate_confusion_matrix(y_true, y_pred)

    # We can see the problem.
    # Those that are classified negative are generally negative
    # Those that are classified positive are generally positive
    # But most reviews have been classified as neutral, even many
    # that were positive os negative



if __name__ == "__main__":
    main()

import altair as alt
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

LARGE_DATASET_SIZE = 100000
RANDOM_SEED = 42

data_root = '../data'
data_path = f'{data_root}/Video_Games_5.json'
small_corpus = f'{data_root}/small_corpus.csv'
large_corpus = f'{data_root}/large_corpus.csv'

chart_root = '../charts'
original_chart = f'{chart_root}/chart.html'
small_chart = f'{chart_root}/small_chart.html'
large_chart = f'{chart_root}/large_chart.html'

alt.data_transformers.disable_max_rows()


# Data format

# "overall" - score. Integer [1..5]
# "verified" - Has the review been verified. Boolean
# "reviewerID" - Unique identifier for the review. String.
# "asin" - Unknown. String
# "reviewTime" - Date of review. Date format "MM DD, YYYY"
# "reviewerName" - Name of the reviewer. String
# "reviewText" - The review. String
# "summary" - Summary of the review. String.
# "unixReviewTime" - Timestamp for review. Unix Epoch format


def create_chart(df: pd.DataFrame, filename: str):
    chart = alt.Chart(df).mark_bar().encode(x='overall', y='count()')
    chart.save(filename)


def load_data_frame() -> pd.DataFrame:
    # Pandas now supports NDJson out of the box
    df = pd.read_json(data_path, orient='records', lines=True)
    create_chart(df, original_chart)
    return df


def corpus_to_csv(df: pd.DataFrame, filename: str):
    df = df[['reviewText', 'overall']]
    df = df.rename({'reviewText': 'reviews', 'overall': 'ratings'})
    df.to_csv(filename, index=False)


def create_small_data_frame(df):
    strategy = {1: 1500, 2: 500, 3: 500, 4: 500, 5: 1500}
    under_sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=RANDOM_SEED)
    x, _ = under_sampler.fit_resample(df, df['overall'])
    create_chart(x, small_chart)
    corpus_to_csv(x, small_corpus)


def create_large_data_frame(df):
    np.random.seed(seed=RANDOM_SEED)
    random_indexes = np.random.randint(0, len(df), LARGE_DATASET_SIZE)
    large_df = df.iloc[random_indexes]
    create_chart(large_df, large_chart)
    corpus_to_csv(large_df, large_corpus)


def main():
    print("Loading data...")
    df = load_data_frame()

    print("Creating small corpus..")
    create_small_data_frame(df)

    print("Creating large corpus..")
    create_large_data_frame(df)

    print("All done")


if __name__ == "__main__":
    main()

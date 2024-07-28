import altair as alt
import nltk
from nltk.corpus import opinion_lexicon
from nltk.sentiment import util
import numpy as np
import pandas as pd
import scipy.stats as stats


data_root = '../data'
data_path = f'{data_root}/Video_Games_5.json'
small_corpus = f'{data_root}/small_corpus.csv'
small_corpus_scored = f'{data_root}/small_corpus_scored.csv'

chart_root = '../charts/part-2'
ratings_distribution = f'{chart_root}/ratings_distribution.html'
score_distribution = f'{chart_root}/score_distribution.html'
score_distribution_no_negation = f'{chart_root}/score_distribution_no_negation.html'
ratings_score_correlation = f'{chart_root}/ratings_score_correlation.html'
ratings_score_correlation_no_negation = f'{chart_root}/ratings_score_correlation_no_negation.html'


negative_words = set(opinion_lexicon.negative())
positive_words = set(opinion_lexicon.positive())

alt.data_transformers.disable_max_rows()


def load_datasets():
    nltk.download('opinion_lexicon')
    nltk.download('punkt')


def score_sentence(sentence, handle_negation):
    sentence_tokenised = nltk.tokenize.word_tokenize(sentence)
    if handle_negation:
        sentence_tokenised = util.mark_negation(sentence_tokenised, shallow=True)

    score = 0.0
    for word in sentence_tokenised:
        word = word.lower()
        if word in positive_words:
            score += 1.0
        elif word in negative_words:
            score -= 1.0

    return score / len(sentence_tokenised) if len(sentence_tokenised) != 0 else 0


def score_review(review, handle_negation):
    tokenised = nltk.tokenize.sent_tokenize(review)
    score = 0.0
    for sentence in tokenised:
        score += score_sentence(sentence, handle_negation)
    return score / len(tokenised)


def chart_ratings_distribution(df):
    chart = alt.Chart(df).mark_bar().encode(x='ratings', y='count()')
    chart.save(ratings_distribution)


def chart_sentiment_score_distribution(df, handle_negation):
    values, buckets = np.histogram(df['score'], 5)
    chart_df = pd.DataFrame({'buckets': buckets[:-1], 'values': values})
    chart = alt.Chart(chart_df).mark_bar().encode(x='buckets', y='values')
    chart.save(score_distribution if handle_negation else score_distribution_no_negation)


def chart_ratings_sentiment_score_correlation(df, handle_negation):
    chart = alt.Chart(df).mark_point().encode(x='ratings', y="score")
    chart.save(ratings_score_correlation if handle_negation else ratings_score_correlation_no_negation)


def calculate_ratings_bucket_sizes(df):
    # Returns the size of each ratings bucket from the lowest rating
    # to the highest rating
    return (df.groupby('ratings').size().reset_index(name='count')
            .sort_values(by='ratings', ascending=True)
            ['count'].to_numpy())


def add_score_buckets(df, bucket_sizes):
    bucket_index = 0
    bucket_count = 0

    for index, row in df.sort_values(by='score', ascending=True).iterrows():
        df.at[index, 'score_bucket'] = bucket_index + 1
        bucket_count += 1
        if bucket_count >= bucket_sizes[bucket_index]:
            bucket_count = 0
            bucket_index += 1


def create_contingency_table(df):
    # https://towardsdatascience.com/chi-square-test-for-correlation-test-in-details-manual-and-python-implementation-472ae5c4b15f
    contingency = pd.crosstab(df['ratings'], df["score_bucket"])
    return contingency.values


def print_contingency_table(contingency_table):
    # Buckets by rating and score_bucket - showing number of entries in each bucket
    # Useful for seeing how correlated they are
    print()
    print()
    print("Contingency table")
    print()
    print(f'Score   ', end='')
    for score_bucket in range(5):
        print(f'{score_bucket:>{5}} ', end='')
    print()
    print('--------+---------------------------------')

    for rating in range(len(contingency_table)):
        print(f'Rating {rating}| ', end='')
        for score_bucket in range(len(contingency_table[rating])):
            value = contingency_table[rating][score_bucket]
            print(f'{value:>{5}} ', end='')

        print()


def sum_diagonals(table, offset):
    indexes = range(len(table) - offset)
    part_1 = sum([table[i, offset + i] for i in indexes])
    part_2 = sum([table[i + offset, i] for i in indexes])
    return part_1 + (part_2 if offset != 0  else 0)


def spearmans_rho(df):
    # Null hypothesis is they are independent
    # p value - probability result this far from the mean occurred under h0
    # Statistic - -1 negative correlation, 0 no correlation, 1 positive correlation
    x = df['ratings'].to_numpy()
    y = df['score'].to_numpy()
    return stats.spearmanr(x, y)


def contingency_table_statistics(contingency_table):
    # Chi squared test
    # Null hypothesis: The two variables are independent
    # p-value is the probability of observing a test statistic as extreme as the one observed
    # given that the null hypothesis is true
    # if p-value is less than  say 0.05, we reject the null hypothesis and conclude that the two variables are dependent
    # Distribution independent - presumably relies on the law of large numbers

    numbers_placed = [sum_diagonals(contingency_table, i) for i in range(len(contingency_table))]
    total_reviews = sum(numbers_placed)
    chi_squared, p, degrees_of_freedom, _ = stats.chi2_contingency(contingency_table)
    return numbers_placed, total_reviews, chi_squared, p, degrees_of_freedom


def print_contingency_table_stats(ct_stats, rho):
    print()
    print()
    print("Placement errors")
    numbers_placed, total_reviews, chi_squared, p, degrees_of_freedom = ct_stats
    percent = 100 * numbers_placed[0] / total_reviews
    print(f'Correct: {numbers_placed[0]}, {percent:.4}%')
    for i in range(1, len(numbers_placed)):
        percent = 100*numbers_placed[i]/total_reviews
        print(f'Placement error {i}: {numbers_placed[i]}, {percent:.4}%')
    print()
    print('Chi squared measure of independence')
    print(f'Probability that rating and score are independent (chi squared p-value) {p}')
    print(f'Chi squared statistic: {chi_squared:.8}')
    print(f'Degrees of freedom: {degrees_of_freedom}')
    print()
    print(f'Rho: {rho}')


def load_ds_and_score(df, handle_negation):
    df = df.copy(deep=True)
    df['score'] = df.apply(lambda row: score_review(row['reviews'], handle_negation), axis=1)
    bucket_sizes = calculate_ratings_bucket_sizes(df)
    add_score_buckets(df, bucket_sizes)

    # Show distribution and correlation graphically
    chart_sentiment_score_distribution(df, handle_negation)
    chart_ratings_sentiment_score_correlation(df, handle_negation)

    # Correlation measures
    contingency_table = create_contingency_table(df)
    print_contingency_table(contingency_table)
    ct_stats = contingency_table_statistics(contingency_table)
    rho = spearmans_rho(df)
    print_contingency_table_stats(ct_stats, rho)

    # We can conclude that ratings and score are not independent but that
    # score cannot be used as a proxy for ratings

    # Results without negation handled
    #  Contingency table

    # Score       0     1     2     3     4
    # --------+---------------------------------
    # Rating 0|   905   191   129   102   173
    # Rating 1|   234    69    73    50    74
    # Rating 2|   153    76    94    58   119
    # Rating 3|    67    62    84    86   201
    # Rating 4|   141   102   120   204   933

    # Placement errors
    # Correct: 2087, 46.38%
    # Placement error 1: 1121, 24.91%
    # Placement error 2: 633, 14.07%
    # Placement error 3: 345, 7.667%
    # Placement error 4: 314, 6.978%

    # Chi squared measure of independence
    # Probability that rating and score are independent (chi squared p-value) 9.968732932354007e-309
    # Chi squared statistic: 1493.992
    # Degrees of freedom: 16

    # Rho: SignificanceResult(statistic=0.5659496840395417, pvalue=0.0)

    # Results with negation handled
    # Contingency table
    #
    # Score       0     1     2     3     4
    # --------+---------------------------------
    # Rating 0|   918   218   146   105   113
    # Rating 1|   245    55    82    56    62
    # Rating 2|   148    74   102    66   110
    # Rating 3|    72    33    80    95   220
    # Rating 4|   117   120    90   178   995

    # Placement errors
    # Correct: 2165, 48.11%
    # Placement error 1: 1163, 25.84%
    # Placement error 2: 583, 12.96%
    # Placement error 3: 359, 7.978%
    # Placement error 4: 230, 5.111%

    # Chi squared measure of independence
    # Probability that rating and score are independent (chi squared p-value) 0.0
    # Chi squared statistic: 1824.88
    # Degrees of freedom: 16
    #
    # Rho: SignificanceResult(statistic=0.6166537072839097, pvalue=0.0)

    return df


def save_corpus(df, filename):
    df.to_csv(filename, index=False)

def worst_offenders():
    # s = "A brand new huge open world to explore and the entire world from the first game with new enemies and a new outlook on the worlds around Gravity Rush."
    # enemies makes it negative

    # s = "Mario and Sonic never disappoint."
    # negation of disappoint

    # s = "Great little limited edition! and cheap too"
    # cheap is viewed as negative - but here it is positive
    # Great is not picked up - due to case
    # Likewise limited is viewed as negative, but limited edition is positive

    # s = "Good but bad idea for an online-only game."
    # Bad is negative. Good is not recognised due to case
    s = "Item arrived on time with no issues!"
    # Issues negative, negation not picked up

    print(score_review(s, handle_negation=False))


def main():
    # load_datasets()
    df = pd.read_csv(small_corpus)
    chart_ratings_distribution(df)

    load_ds_and_score(df, handle_negation=False)
    df = load_ds_and_score(df, handle_negation=True)
    # save_corpus(df, small_corpus_scored)


if __name__ == "__main__":
    main()
    # worst_offenders()




